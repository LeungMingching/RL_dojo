import gymnasium as gym

from stable_baselines3 import SAC


def train():
    env = gym.make("Pendulum-v1", render_mode="human", g=9.81)

    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000, log_interval=4)
    model.save("saved/model/sac_pendulum")


if __name__ == "__main__":
    train()