"""
A simple implementation of Tabular Q-Learning algorithm.
"""

import yaml
import gymnasium as gym
from agent import Agent
from trainer import Trainer


def main(config: dict):
    env = gym.make(**config["env"])

    ag = Agent(
        config=config["agent"],
        observation_space=env.observation_space,
        action_space=env.action_space
    )

    trainer = Trainer(env=env, agent=ag, config=config["trainer"])
    trainer.train()


if __name__ == "__main__":
    with open("config/config.yaml", "r", encoding="UTF-8") as f:
        config = yaml.safe_load(f)

    main(config)
