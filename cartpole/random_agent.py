import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
rewards = []

for episode in range(10):

    obs, _ = env.reset()
    total_reward = 0
    
    while True:
        action = env.action_space.sample()
        obs, reward, done, truncated, _ = env.step(action)

        total_reward += reward
        rewards.append(total_reward)

        if done or truncated:
            break


print(f"Average reward: {sum(rewards)/len(rewards)}")
env.close()