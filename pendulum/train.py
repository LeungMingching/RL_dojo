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

    # Load model if a preload path is specified
    if config.model.preload_path:
        print(f"Loading model from {config.model.preload_path}")
        model = SAC.load(config.model.preload_path, env=env)
    else:
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=config.tensorboard_log,
            device=config.model.device,
        )
        
    # Train the model
    model.learn(
        total_timesteps=config.total_timesteps,
        log_interval=config.log_interval,
        progress_bar=True
    )
    model.save(config.model.save_path)


if __name__ == "__main__":
    main()