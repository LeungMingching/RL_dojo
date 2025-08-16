import hydra
import gymnasium as gym

from stable_baselines3 import SAC


@hydra.main(version_base=None, config_path="config", config_name="evaluate")
def main(config):
    env = gym.make(
        config.env.name,
        render_mode=config.env.render_mode,
        g=config.env.gravity
    )

    # Load model
    model = SAC.load(config.model_path)

    # Evaluate
    total_reward = 0.0

    observation, info = env.reset()
    while True:
        
        try:
            action = model.predict(observation, deterministic=True)[0]
            observation, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            if terminated or truncated:
                observation, info = env.reset()

                print(f"Total reward: {total_reward}")

                total_reward = 0.0
        except KeyboardInterrupt:
            print("Exiting...")
            break


if __name__ == "__main__":
    main()