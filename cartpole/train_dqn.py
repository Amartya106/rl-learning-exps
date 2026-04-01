import torch.optim as optim
import torch
import gymnasium as gym
from cartpole.dqn import DQN
from common.replay_buffer import ReplayBuffer
import random
import numpy as np

env = gym.make("CartPole-v1", render_mode = 'human')

model = DQN()

optimizer = optim.Adam(model.parameters(), lr = 1e-3)
loss_fn = torch.nn.MSELoss()
gamma = 0.99

buffer = ReplayBuffer(10000)
epsilon = 0.99
decay_factor = 0.995
min_epsilon = 0.5

episode_num = 200

#Training Loop
for episode in range(episode_num):
    obs, _ = env.reset()
    total_reward = 0

    #Epsilon Greedy
    while True:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                q_values = model(state)
            action = torch.argmax(q_values).item()
    
        next_obs, reward, done, truncated, _ = env.step(action)
        done_flag = done or truncated

        #store experience
        buffer.add(obs, action, reward, next_obs, done_flag)

        obs = next_obs
        total_reward += reward

        if buffer.size() > 64:
            batch = buffer.sample(64)

            states      = torch.from_numpy(np.array([b[0] for b in batch])).float()
            actions     = torch.tensor([b[1] for b in batch], dtype=torch.long)
            rewards     = torch.tensor([b[2] for b in batch], dtype=torch.float32)
            next_states = torch.from_numpy(np.array([b[3] for b in batch])).float()
            dones       = torch.tensor([b[4] for b in batch], dtype=torch.float32)

            q_values = model(states)
            q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()

            with torch.no_grad():
                next_q_values = model(next_states).max(1)[0]
                targets = rewards + gamma* next_q_values * (1-dones)
            
            loss = loss_fn(q_values, targets)

            #Backpropogation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if done_flag:
            break

    epsilon = max(min_epsilon, epsilon*decay_factor)

    print(f"Episode: {episode}, Reward: {total_reward}, Epsilon {epsilon:.4f}")

env.close()


