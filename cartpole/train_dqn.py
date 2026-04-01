import torch.optim as optim
import torch

from cartpole.dqn import DQN
from common.replay_buffer import ReplayBuffer


model = DQN()

optimizer = optim.Adam(model.parameters(), lr = 1e-3)
loss_fn = torch.nn.MSELoss()
gamma = 0.99

buffer = ReplayBuffer(10000)