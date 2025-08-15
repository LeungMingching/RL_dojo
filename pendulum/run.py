import gymnasium as gym


def main():
    # Create the Pendulum environment
    env = gym.make("Pendulum-v1", render_mode="human", g=9.81)

    # Reset environment to start a new episode
    observation, info = env.reset()

    print(f"Starting observation: {observation}")

    episode_over = False
    total_reward = 0.0

    while not episode_over:

        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        episode_over = terminated or truncated

    print(f"Episode finished! Total reward: {total_reward}")
    env.close()



if __name__ == "__main__":
    main()