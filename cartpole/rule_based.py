import gymnasium as gym

env = gym.make("CartPole-v1", render_mode='human')
rewards = []

for episode in range(10):
    obs, _ = env.reset()
    # for cartpole obs is (x, velocity, angle, angular_velocity)
    total_reward = 0

    while True:
        if obs[2] > 0:
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