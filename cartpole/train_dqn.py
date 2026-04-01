import torch.optim as optim
import torch
import gymnasium as gym
from cartpole.dqn import DQN
from common.replay_buffer import ReplayBuffer
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1")

model = DQN()
target_model = DQN()
target_model.load_state_dict(model.state_dict())
target_model.eval()

optimizer = optim.Adam(model.parameters(), lr = 1e-4)
loss_fn = torch.nn.SmoothL1Loss()
gamma = 0.99

buffer = ReplayBuffer(50000)
epsilon = 1
decay_factor = 0.997
min_epsilon = 0.01

episode_num = 2000

reward_history = []
avg_rewards = []
window = deque(maxlen=100)

total_steps = 0
best_avg = -float('inf')


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

        if buffer.size() >= 128:
            batch = buffer.sample(128)

            states      = torch.from_numpy(np.array([b[0] for b in batch])).float()
            actions     = torch.tensor([b[1] for b in batch], dtype=torch.long)
            rewards     = torch.tensor([b[2] for b in batch], dtype=torch.float32)
            next_states = torch.from_numpy(np.array([b[3] for b in batch])).float()
            dones       = torch.tensor([b[4] for b in batch], dtype=torch.float32)

            q_values = model(states)
            q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_q_values = target_model(next_states).max(1)[0]
                targets = rewards + gamma* next_q_values * (1-dones)
            
            loss = loss_fn(q_values, targets)

            #Backpropogation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            total_steps +=1 
            if total_steps %1000 == 0:
                target_model.load_state_dict(model.state_dict())
        
        if done_flag:
            break

    epsilon = max(min_epsilon, epsilon*decay_factor)

    reward_history.append(total_reward)
    window.append(total_reward)
    avg_reward = sum(window)/len(window)
    avg_rewards.append(avg_reward)

    

    if avg_reward > best_avg:
        best_avg = avg_reward
        torch.save(model.state_dict(), "results/dqn_cartpole_best.pt")
    
    if avg_reward >= 475:
        print(f"Solved at episode {episode}!")
        torch.save(model.state_dict(), "results/dqn_cartpole_final.pt")
        break

    print(f"Episode: {episode}, Reward: {total_reward}, Epsilon {epsilon:.4f}, Avg(100): {avg_reward:.2f}")

plt.plot(reward_history, label="Reward")
plt.plot(avg_rewards, label="Avg(100)")
plt.legend()
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN Training")
plt.savefig("results/training_plot_cp.png")

plt.show()
torch.save(model.state_dict(), "results/dqn_cartpole.pt")

env.close()


