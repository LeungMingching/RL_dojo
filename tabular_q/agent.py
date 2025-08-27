import numpy as np


class Agent:
    def __init__(self, config, observation_space, action_space):
        self.config = config
        self.learning_rate = config.get("learning_rate", 0.1)
        self.discount_factor = config.get("discount_factor", 0.99)
        self.exporation_rate = config.get("exploration_rate", 0.1)

        self.observation_space = observation_space
        self.action_space = action_space
        self.q_table = self._initialize_q_table()

        self.last_observation = None
        self.last_action = None

    def predict(self, observation, explore=False):

        if explore and np.random.rand() < self.exporation_rate:
            action = self.action_space.sample()
        else:
            q_values = self.q_table[observation]
            max_value = q_values.max()
            max_actions = np.flatnonzero(q_values == max_value)
            action = np.random.choice(max_actions)

        self.last_observation = observation
        self.last_action = action

        return action

    def _initialize_q_table(self):
        observation_size = self.observation_space.n
        action_size = self.action_space.n
        return np.zeros((observation_size, action_size))

    def update(self, observation, reward, done):
        self.q_table[self.last_observation, self.last_action] += \
            self.learning_rate * (
                reward +
                self.discount_factor * (0 if done else self.q_table[observation].max()) -
                self.q_table[self.last_observation, self.last_action]
            )
        
        print(self.q_table)