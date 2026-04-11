import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
from robosuite.wrappers import GymWrapper

from td3_torch import Agent


def _joint_velocity_config():
    """Composite controller config with JOINT_VELOCITY for the Panda arm.

    robosuite 1.5 uses composite controllers. Build the config dict directly —
    do NOT use refactor_composite_controller_config(); it triggers a read-only
    property setter error on JointVelocityController.
    """
    return {
        "type": "BASIC",
        "body_parts": {
            "right": {
                "type": "JOINT_VELOCITY",
                "input_max": 1.0,
                "input_min": -1.0,
                "output_max": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                "output_min": [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0],
                "kp": 100,
                "velocity_limits": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            },
            "gripper": {"type": "GRIP"},
        },
    }


def make_env(has_renderer=False):
    """Create the robosuite Door environment wrapped in the Gym/Gymnasium interface."""
    env = suite.make(
        'Door',
        robots='Panda',
        controller_configs=_joint_velocity_config(),
        has_renderer=has_renderer,
        use_camera_obs=False,   # no pixel observations; state-based only
        horizon=300,            # episode length sweet spot for this task
        reward_shaping=True,    # dense rewards make learning much faster
        control_freq=20,
    )
    return GymWrapper(env)


def reset_env(env):
    """Handle both old gym (returns obs) and new gymnasium (returns obs, info)."""
    result = env.reset()
    if isinstance(result, tuple):
        return result[0]
    return result


def step_env(env, action):
    """Handle both old gym (4-tuple) and new gymnasium (5-tuple) step returns."""
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = terminated or truncated
    else:
        obs, reward, done, info = result
    return obs, reward, done, info


if __name__ == '__main__':
    env = make_env()

    # ── Hyper-parameters ──────────────────────────────────────────────────────
    actor_lr    = 0.001
    critic_lr   = 0.001
    batch_size  = 128
    layer1_size = 256
    layer2_size = 128

    agent = Agent(
        actor_learning_rate=actor_lr,
        critic_learning_rate=critic_lr,
        tau=0.005,
        input_dims=env.observation_space.shape,
        env=env,
        n_actions=env.action_space.shape[0],
        layer1_size=layer1_size,
        layer2_size=layer2_size,
        batch_size=batch_size,
    )

    writer = SummaryWriter('logs')
    n_games = 10_000
    best_score = 0

    # Label includes hyper-parameters so TensorBoard graphs are self-documenting
    run_label = (f'actor_lr={actor_lr}-critic_lr={critic_lr}'
                 f'-bs={batch_size}-l1={layer1_size}-l2={layer2_size}')

    agent.load_models()

    for episode in range(n_games):
        observation = reset_env(env)
        done = False
        score = 0.0

        while not done:
            action = agent.choose_action(observation)
            next_observation, reward, done, _ = step_env(env, action)
            score += reward
            agent.remember(observation, action, reward, next_observation, done)
            agent.learn()
            observation = next_observation   # ← critical: advance the state

        writer.add_scalar(f'score/{run_label}', score, global_step=episode)

        if episode % 10 == 0:
            agent.save_models()

        print(f'Episode {episode:5d}  Score: {score:8.2f}')

    writer.close()
    env.close()
