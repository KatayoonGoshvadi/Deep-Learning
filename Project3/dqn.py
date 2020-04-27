import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pdb

# The starter code follows the tutorial: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# we recommend you going through the tutorial before implement DQN algorithm


# define environment, please don't change 
env = gym.make('CartPole-v1')

# define transition tuple
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    """
    define replay buffer class
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    """
    build your DQN model:
    given the state, output the possiblity of actions
    """
    def __init__(self, in_dim, out_dim):
        """
        in_dim: dimension of states
        out_dim: dimension of actions
        """
        super(DQN, self).__init__()
        # build your model here
        self.l1 = nn.Linear(in_dim, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 64)
        self.l4 = nn.Linear(64, 2)

    def forward(self, x):
        # forward pass
        # pdb.set_trace()
        l1  = F.relu(self.l1(x))
        l2  = F.relu(self.l2(l1))
        l3  = F.relu(self.l3(l2))
        out = self.l4(l3)
        return out.to(device)

    # def load_state_dict(weights):
    #   self.parameters = weights


# hyper parameters you can play with
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 500
TARGET_UPDATE = 10
MEMORY_CAPACITY = 10000

n_actions = env.action_space.n
n_states = env.observation_space.shape[0]

policy_net = DQN(n_states, n_actions)
target_net = DQN(n_states, n_actions)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters())
memory = ReplayMemory(MEMORY_CAPACITY)

criterion = nn.MSELoss()

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


steps_done = 0

def select_action(state, Testing):
    # given state, return the action with highest probability on the prediction of DQN model
    # you are recommended to also implement a soft-greedy here
    global steps_done
    sample = random.random()
    EPS = EPS_END + (EPS_START - EPS_END) * \
    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    # print(EPS)

    if  np.random.random_sample() < EPS and not Testing:
        a = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
    else:
        with torch.no_grad():
            a = policy_net(state).max(1)[1].view(1, 1)
    return a

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    # pdb.set_trace()
    state_batch  = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


num_episodes = 100
episode_durations = []
for i_episode in range(num_episodes):
    # Initialize the environment and state
    state = env.reset()
    state = torch.from_numpy(state).float().view(1, -1)
    for t in count():
        # Select and perform an action
        action = select_action(state, False)
        new_state, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # # Observe new state
        if not done:
            next_state = torch.from_numpy(new_state).float().view(1, -1)
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            print("Episode: {}, duration: {}".format(i_episode, t+1))
            break
        # steps_done++
    
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

# saving the model
torch.save({
'policy': policy_net.state_dict(),
'target': target_net.state_dict(),
'optimizer_state_dict': optimizer.state_dict(),
}, './p2_model.ckpt')

# plot time duration
plt.figure()
plt.plot(np.arange(len(episode_durations)), episode_durations)
plt.show()


# visualize 
for i in range(10):
    state = env.reset()
    state = torch.from_numpy(state).float().view(1, -1)
    for t in count():
        env.render()

        # Select and perform an action
        action = select_action(state, True)
        new_state, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward])

        # Observe new state
        if not done:
            next_state = torch.from_numpy(new_state).float().view(1, -1)
        else:
            next_state = None

        # Move to the next state
        state = next_state

        if done:
            episode_durations.append(t + 1)
            print("Duration:", t+1)
            break

env.close()





