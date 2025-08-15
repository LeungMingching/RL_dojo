import gymnasium as gym

from stable_baselines3 import SAC


def main():
    env = gym.make("Pendulum-v1", render_mode="human", g=9.81)

    # Load model
    model = SAC.load("saved/model/sac_pendulum")

    # Evaluate
    total_reward = 0.0

    observation, info = env.reset()
    while True:
        action = model.predict(observation, deterministic=True)[0]
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        if terminated or truncated:
            observation, info = env.reset()

            print(f"Total reward: {total_reward}")

            total_reward = 0.0


if __name__ == "__main__":
    main()