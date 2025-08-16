import hydra
import gymnasium as gym

from stable_baselines3 import SAC


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(config):
    env = gym.make(
        config.env.name,
        render_mode=config.env.render_mode,
        g=config.env.gravity
    )

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=config.tensorboard_log,
    )
    model.learn(
        total_timesteps=config.total_timesteps, log_interval=config.log_interval)
    model.save(config.model_path)


if __name__ == "__main__":
    main()