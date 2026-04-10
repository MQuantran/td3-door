import os
import numpy as np
import torch as T
import torch.nn.functional as F

from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork


class Agent:
    def __init__(self, actor_learning_rate, critic_learning_rate, input_dims,
                 env, tau=0.005, gamma=0.99, update_actor_interval=2,
                 warmup=1000, n_actions=2, max_size=1_000_000,
                 layer1_size=256, layer2_size=128, batch_size=100, noise=0.1):

        self.gamma = gamma
        self.tau = tau
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval

        chkpt_dir = 'tmp/td3'
        os.makedirs(chkpt_dir, exist_ok=True)

        # ── Six networks: actor + 2 critics + their three target copies ────────
        self.actor = ActorNetwork(input_dims, fc1_dims=layer1_size,
                                  fc2_dims=layer2_size, n_actions=n_actions,
                                  name='actor', chkpt_dir=chkpt_dir,
                                  learning_rate=actor_learning_rate)

        self.critic_1 = CriticNetwork(input_dims, n_actions,
                                      fc1_dims=layer1_size, fc2_dims=layer2_size,
                                      name='critic_1', chkpt_dir=chkpt_dir,
                                      learning_rate=critic_learning_rate)

        self.critic_2 = CriticNetwork(input_dims, n_actions,
                                      fc1_dims=layer1_size, fc2_dims=layer2_size,
                                      name='critic_2', chkpt_dir=chkpt_dir,
                                      learning_rate=critic_learning_rate)

        self.target_actor = ActorNetwork(input_dims, fc1_dims=layer1_size,
                                         fc2_dims=layer2_size, n_actions=n_actions,
                                         name='target_actor', chkpt_dir=chkpt_dir,
                                         learning_rate=actor_learning_rate)

        self.target_critic_1 = CriticNetwork(input_dims, n_actions,
                                             fc1_dims=layer1_size, fc2_dims=layer2_size,
                                             name='target_critic_1', chkpt_dir=chkpt_dir,
                                             learning_rate=critic_learning_rate)

        self.target_critic_2 = CriticNetwork(input_dims, n_actions,
                                             fc1_dims=layer1_size, fc2_dims=layer2_size,
                                             name='target_critic_2', chkpt_dir=chkpt_dir,
                                             learning_rate=critic_learning_rate)

        self.noise = noise
        # tau=1 on first call does a full hard copy of weights into target nets
        self.update_network_parameters(tau=1)

    # ── Action selection ───────────────────────────────────────────────────────

    def choose_action(self, observation, validation=False):
        if self.time_step < self.warmup and not validation:
            # Pure random exploration during warm-up
            mu = T.tensor(np.random.normal(scale=self.noise,
                                           size=(self.n_actions,)),
                          dtype=T.float).to(self.actor.device)
        else:
            state = T.tensor(observation, dtype=T.float).to(self.actor.device)
            mu = self.actor.forward(state)

        # TD3 exploration: add clipped Gaussian noise to the actor output
        noise = T.tensor(np.random.normal(scale=self.noise,
                                          size=mu.shape),
                         dtype=T.float).to(self.actor.device)
        mu_prime = mu + noise
        mu_prime = T.clamp(mu_prime,
                           float(self.min_action[0]),
                           float(self.max_action[0]))

        self.time_step += 1
        return mu_prime.cpu().detach().numpy()

    # ── Memory ─────────────────────────────────────────────────────────────────

    def remember(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    # ── Learning step ──────────────────────────────────────────────────────────

    def learn(self):
        if self.memory.mem_cntr < self.batch_size * 10:
            return

        state, action, reward, next_state, done = \
            self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        done = T.tensor(done).to(self.critic_1.device)
        next_state = T.tensor(next_state, dtype=T.float).to(self.critic_1.device)
        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)

        # ── Critic update ──────────────────────────────────────────────────────
        with T.no_grad():
            # Target policy smoothing: add clipped noise to target actions
            target_actions = self.target_actor.forward(next_state)
            smoothing_noise = T.clamp(
                T.tensor(np.random.normal(scale=0.2, size=target_actions.shape),
                         dtype=T.float).to(self.actor.device),
                -0.5, 0.5)
            target_actions = T.clamp(target_actions + smoothing_noise,
                                     float(self.min_action[0]),
                                     float(self.max_action[0]))

            # Clipped double-Q: use minimum of two target critics
            next_q1 = self.target_critic_1.forward(next_state, target_actions)
            next_q2 = self.target_critic_2.forward(next_state, target_actions)

            next_q1 = next_q1.view(-1)
            next_q2 = next_q2.view(-1)
            next_q1[done] = 0.0   # terminal states have no future value
            next_q2[done] = 0.0

            next_critic_value = T.min(next_q1, next_q2)
            target = reward + self.gamma * next_critic_value
            target = target.view(self.batch_size, 1)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q1 = self.critic_1.forward(state, action)
        q2 = self.critic_2.forward(state, action)

        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_step_cntr += 1

        # ── Delayed actor update ───────────────────────────────────────────────
        # Actor is updated less frequently than critics (TD3 "twin delayed" part)
        if self.learn_step_cntr % self.update_actor_iter != 0:
            return

        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state))
        # Gradient *ascent* on Q → negate for gradient descent optimizer
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    # ── Polyak averaging of target network weights ─────────────────────────────

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = dict(self.actor.named_parameters())
        critic_1_params = dict(self.critic_1.named_parameters())
        critic_2_params = dict(self.critic_2.named_parameters())
        target_actor_params = dict(self.target_actor.named_parameters())
        target_critic_1_params = dict(self.target_critic_1.named_parameters())
        target_critic_2_params = dict(self.target_critic_2.named_parameters())

        for name in critic_1_params:
            critic_1_params[name] = (tau * critic_1_params[name].clone()
                                     + (1 - tau) * target_critic_1_params[name].clone())
        for name in critic_2_params:
            critic_2_params[name] = (tau * critic_2_params[name].clone()
                                     + (1 - tau) * target_critic_2_params[name].clone())
        for name in actor_params:
            actor_params[name] = (tau * actor_params[name].clone()
                                  + (1 - tau) * target_actor_params[name].clone())

        self.target_critic_1.load_state_dict(critic_1_params)
        self.target_critic_2.load_state_dict(critic_2_params)
        self.target_actor.load_state_dict(actor_params)

    # ── Persistence ────────────────────────────────────────────────────────────

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models(self):
        try:
            self.actor.load_checkpoint()
            self.target_actor.load_checkpoint()
            self.critic_1.load_checkpoint()
            self.critic_2.load_checkpoint()
            self.target_critic_1.load_checkpoint()
            self.target_critic_2.load_checkpoint()
            print('Successfully loaded models.')
        except Exception:
            print('Failed to load models. Starting from scratch.')
