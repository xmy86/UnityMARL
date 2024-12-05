import numpy as np
from gym.spaces import Box
from xuance.environment import RawEnvironment

class MyNewEnv(RawEnvironment):
    def __init__(self, env_config):
        super(MyNewEnv, self).__init__()
        self.env_id = env_config.env_id  # The environment id.
        self.observation_space = Box(-np.inf, np.inf, shape=[18, ])  # Define observation space.
        self.action_space = Box(-np.inf, np.inf, shape=[5, ])  # Define action space. In this example, the action space is continuous.
        self.max_episode_steps = 32  # The max episode length.
        self._current_step = 0  # The count of steps of current episode.

    def reset(self, **kwargs):  # Reset your environment.
        self._current_step = 0
        return self.observation_space.sample(), {}

    def step(self, action):  # Run a step with an action.
        self._current_step += 1
        observation = self.observation_space.sample()
        rewards = np.random.random()
        terminated = False
        truncated = False if self._current_step < self.max_episode_steps else True
        info = {}
        return observation, rewards, terminated, truncated, info

    def render(self, *args, **kwargs):  # Render your environment and return an image if the render_mode is "rgb_array".
        return np.ones([64, 64, 64])

    def close(self):  # Close your environment.
        return
    
import argparse
from xuance.common import get_configs
configs_dict = get_configs(file_dir="toturial/create_env/config.yaml")
configs = argparse.Namespace(**configs_dict)

from xuance.environment import REGISTRY_ENV
REGISTRY_ENV[configs.env_name] = MyNewEnv

from xuance.environment import make_envs
from xuance.torch.agents import DDPG_Agent

envs = make_envs(configs)  # Make parallel environments.
Agent = DDPG_Agent(config=configs, envs=envs)  # Create a DDPG agent from XuanCe.
Agent.train(configs.running_steps // configs.parallels)  # Train the model for numerous steps.
Agent.save_model("final_train_model.pth")  # Save the model to model_dir.
Agent.finish()  # Finish the training.