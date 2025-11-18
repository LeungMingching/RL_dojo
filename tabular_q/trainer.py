from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics

import os


class Trainer:
    def __init__(self, env, agent, config):
        self.env = env
        self.agent = agent
        self.config = config

        self.num_train_episodes = config.get("num_train_episodes", 1000)
        self.max_steps_per_episode = config.get("max_steps_per_episode", 100)
        
        self.evaluate_per_episodes = config.get("evaluate_per_episodes", 100)
        self.num_evaluate_episodes = config.get("num_evaluate_episodes", 10)

        self.save_path = config.get("save_path", "saved")
        self.save_policy_per_episodes = config.get("save_policy_per_episodes", 10)

        self.record_env = RecordVideo(
            self.env,
            video_folder=os.path.join(self.save_path, "videos"),
            name_prefix="eval",
            episode_trigger=lambda x: x % self.num_evaluate_episodes == 0
        )
        self.record_env = RecordEpisodeStatistics(self.record_env)

    def train(self):
        for episode in range(self.num_train_episodes):
            self._train_one_episode(episode)

            # Model evaluation
            if self._is_reached_evaluation_episode(episode):
                self.evaluate()
            
            # Model saving
            if self._is_reached_save_episode(episode):
                model_dir = f"{self.save_path}/models"
                os.makedirs(model_dir,exist_ok=True)
                file_path = os.path.join(model_dir, f"train_episode_{episode+1}.npy")
                self.agent.save_policy(file_path)
        
        self.env.close()

    def evaluate(self):
        for episode in range(self.num_evaluate_episodes):
            self._evaluate_one_episode(episode)

        self.record_env.close()
    
    def _train_one_episode(self, episode):
        
        # init
        observation, info = self.env.reset()
        done = False
        step = 0

        while not done:
            action = self.agent.predict(observation)
            observation, reward, terminated, truncated, info = self.env.step(action)

            step += 1

            done = terminated or truncated or self._is_reached_max_steps(step)
            self.agent.update(observation, reward, done)

    def _evaluate_one_episode(self, episode):
        
        # init
        observation, info = self.record_env.reset()
        done = False
        step = 0

        while not done:
            action = self.agent.predict(observation, explore=False)
            observation, reward, terminated, truncated, info = self.record_env.step(action)

            step += 1

            done = terminated or truncated or self._is_reached_max_steps(step)
    
    def _is_reached_max_steps(self, step):
        return step >= self.max_steps_per_episode
    
    def _is_reached_save_episode(self, episode):
        return (episode + 1) % self.save_policy_per_episodes == 0
    
    def _is_reached_evaluation_episode(self, episode):
        return (episode + 1) % self.evaluate_per_episodes == 0