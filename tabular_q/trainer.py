class Trainer:
    def __init__(self, env, agent, config):
        self.env = env
        self.agent = agent
        self.config = config
        self.num_train_episodes = config.get("num_train_episodes", 1000)
        self.max_steps_per_episode = config.get("max_steps_per_episode", 100)
        self.evaluate_per_episodes = config.get("evaluate_per_episodes", 100)

    def train(self):
        for episode in range(self.num_train_episodes):
            self._train_one_episode(episode)
        
        self.env.close()
    
    def evaluate(self, num_eval_step):
        observation, info = self.env.reset()
        for step in range(num_eval_step):
            action = self.agent.predict(observation, explore=False)
            observation, reward, terminated, truncated, info = self.env.step(action)
            self.env.render()

            done = terminated or truncated
            if done:
                observation, info = self.env.reset()

        self.env.close()
    
    def _train_one_episode(self, episode):
        
        # init
        observation, info = self.env.reset()
        done = False
        step = 0

        while not done:
            action = self.agent.predict(observation)
            observation, reward, terminated, truncated, info = self.env.step(action)
            print(observation, reward, terminated, truncated, info)
            self.env.render()

            step += 1

            done = terminated or truncated or self._is_reached_max_steps(step)
            self.agent.update(observation, reward, done)
            if done:
                observation, info = self.env.reset()
    
    def _is_reached_max_steps(self, step):
        return step >= self.max_steps_per_episode