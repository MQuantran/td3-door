import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, fc1_dims=256, fc2_dims=128,
                 name='critic', chkpt_dir='tmp/td3', learning_rate=1e-3):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_td3')

        # Critic takes (state + action) concatenated as input
        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)

        # AdamW with weight decay promotes exploration in the critic
        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate,
                                     weight_decay=0.005)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        print(f'Created {name} network on device: {self.device}')
        self.to(self.device)

    def forward(self, state, action):
        action_value = self.fc1(T.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)
        q1 = self.q1(action_value)
        return q1

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file,
                                    map_location=self.device,
                                    weights_only=True))


class ActorNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims=256, fc2_dims=128,
                 n_actions=2, name='actor', chkpt_dir='tmp/td3',
                 learning_rate=1e-3):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_td3')

        # Actor takes state only and outputs actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.output = nn.Linear(self.fc2_dims, self.n_actions)

        # Plain Adam for the actor (AdamW in both led to insufficient learning)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        print(f'Created {name} network on device: {self.device}')
        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.output(x)
        # tanh squashes output to [-1, 1]; the agent scales it to action bounds
        x = T.tanh(x)
        return x

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file,
                                    map_location=self.device,
                                    weights_only=True))
