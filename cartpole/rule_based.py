import gymnasium as gym

env = gym.make("CartPole-v1", render_mode='human')

obs, _ = env.reset()
# for cartpole obs is (x, velocity, angle, angular_velocity)

total_reward = 0

for _ in range(200):
    if obs[2] > 0:
        action = 1
    else:
        action = 0
    
    obs, reward, done, truncated, _ = env.step(action)
    total_reward+=reward

    if done or truncated:
        break


print(f"Total reward: {total_reward}")
env.close()