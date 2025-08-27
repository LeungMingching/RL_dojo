import yaml
import gymnasium as gym
from agent import Agent


def main(config: dict):
    env = gym.make(**config["env"])
    observation, info = env.reset()

    ag = Agent(
        config=config["agent"],
        observation_space=env.observation_space,
        action_space=env.action_space
    )

    for _ in range(1000):
        action = ag.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        print(observation, reward, terminated, truncated, info)
        env.render()

        done = terminated or truncated
        ag.update(observation, reward, done)
        if done:
            observation, info = env.reset()

    env.close()


if __name__ == "__main__":
    with open("config/config.yaml", "r", encoding="UTF-8") as f:
        config = yaml.safe_load(f)

    main(config)