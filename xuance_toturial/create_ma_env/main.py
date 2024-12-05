import numpy as np
from gym.spaces import Box
from xuance.environment import RawMultiAgentEnv

class MyNewMultiAgentEnv(RawMultiAgentEnv):
    def __init__(self, env_config):
        super(MyNewMultiAgentEnv, self).__init__()
        self.env_id = env_config.env_id
        self.num_agents = 3
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        self.state_space = Box(-np.inf, np.inf, shape=[1, ])
        self.observation_space = {agent: Box(-np.inf, np.inf, shape=[5*40*40, ]) for agent in self.agents}
        self.action_space = {agent: Box(-np.inf, np.inf, shape=[3, ]) for agent in self.agents}
        self.max_episode_steps = 25
        self._current_step = 0

    def get_env_info(self):
        return {'state_space': self.state_space,
                'observation_space': self.observation_space,
                'action_space': self.action_space,
                'agents': self.agents,
                'num_agents': self.num_agents,
                'max_episode_steps': self.max_episode_steps}

    def avail_actions(self):
        return None

    def agent_mask(self):
        """Returns boolean mask variables indicating which agents are currently alive."""
        return {agent: True for agent in self.agents}

    def state(self):
        """Returns the global state of the environment."""
        return self.state_space.sample()

    def reset(self):
        observation = {agent: self.observation_space[agent].sample() for agent in self.agents}
        info = {}
        self._current_step = 0
        return observation, info

    def step(self, action_dict):
        self._current_step += 1
        observation = {agent: self.observation_space[agent].sample() for agent in self.agents}
        rewards = {agent: np.random.random() for agent in self.agents}
        terminated = {agent: False for agent in self.agents}
        truncated = False if self._current_step < self.max_episode_steps else True
        info = {}
        return observation, rewards, terminated, truncated, info

    def render(self, *args, **kwargs):
        return np.ones([64, 64, 64])

    def close(self):
        return
    
import argparse
from xuance.common import get_configs
configs_dict = get_configs(file_dir="toturial/create_ma_env/config.yaml")
configs = argparse.Namespace(**configs_dict)

from xuance.environment import REGISTRY_MULTI_AGENT_ENV
REGISTRY_MULTI_AGENT_ENV[configs.env_name] = MyNewMultiAgentEnv

from xuance.environment import make_envs
from xuance.torch.agents import IPPO_Agents

envs = make_envs(configs)  # Make parallel environments.
Agent = IPPO_Agents(config=configs, envs=envs)  # Create a DDPG agent from XuanCe.
Agent.train(configs.running_steps // configs.parallels)  # Train the model for numerous steps.
Agent.save_model("final_train_model.pth")  # Save the model to model_dir.
Agent.finish()  # Finish the training.