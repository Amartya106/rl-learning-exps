import gymnasium as gym
import random

env = gym.make("CartPole-v1", render_mode='human')
rewards = []

for episodes in range(10):
    obs, _ = env.reset()
    total_reward = 0
    epsilon = 0.2 # 20% randomness

    while True:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            if obs[2]>0:
                action = 1
            else:
                action = 0
    
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        rewards.append(total_reward)

        if done or truncated:
            break

print(f"Average reward: {sum(rewards)/len(rewards)}")
env.close()