import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")

obs, _ = env.reset()
total_reward = 0

for _ in range(500):
    action = env.action_space.sample()
    obs, reward, done, truncated, _ = env.step(action)

    total_reward += reward

    if done or truncated:
        break

print(f"Total reward: {total_reward}")
env.close()