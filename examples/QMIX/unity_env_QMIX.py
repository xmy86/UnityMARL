# /site-packages/mlagents_envs/envs/unity_pettingzoo_base_env.py 
# 135 int32 -> float32

# \site-packages\mlagents_envs\side_channel\side_channel_manager.py
# 45-48 delete

"""
# /site-packages/xuance/environment/vector_envs
# /dummy/dummy_vec_maenv.py
# 27 [np.zeros(space2shape(self.state_space)) for _ in range(self.num_envs)] 
# -> [np.zeros(1) for _ in range(self.num_envs)]

# /site-packages/xuance/torch/policies/deterministic_marl.py
# 38 self.action_space[key].n -> self.action_space[key].shape[0]

# /site-packages/xuance/torch/representations/mlp.py
# 40 self.output_shapes = {'state': (hidden_sizes[-1],)}
# -> self.output_shapes = {'state': (hidden_sizes['fc_hidden_sizes'][-1],)}

# /site-packages/xuance/torch/learners/multi_agent_rl/qmix_learner.py
# 27 self.policy.action_space[k].n -> self.policy.action_space[k].shape[0]
"""

import itertools
import numpy as np
from gym.spaces import Box
from xuance.environment import RawMultiAgentEnv
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_MULTI_AGENT_ENV
from xuance.environment import make_envs
from xuance.torch.agents import QMIX_Agents

# Configuration that needs to be adjusted
config_env_path = "examples/QMIX/QMIX_config.yaml"
configs_dict = get_configs(file_dir=config_env_path)
configs = argparse.Namespace(**configs_dict)
max_episode_steps = 200
worker_id_counter = itertools.count(start=0)

class UnityMultiAgentEnv(RawMultiAgentEnv):
    """
    The implementation of MPE environments, provides a standardized interface for interacting
    with the environments in the context of multi-agent reinforcement learning.

    Parameters:
        config: The configurations of the environment.
    """
    def __init__(self, config):
        super(UnityMultiAgentEnv, self).__init__()
        # Prepare raw environment
        worker_id = next(worker_id_counter)
        self.unity_env = UnityEnvironment(file_name=config.unity_env_path, 
                                          no_graphics=False, 
                                          seed=42,
                                          worker_id=worker_id)
        self.env = UnityParallelEnv(self.unity_env)
        self.env.reset()

        # Set basic attributes
        self.metadata = {'render_modes': None, 
                         'is_parallelizable': True, 
                         'render_fps': 10, 
                         'name': 'Unity_env'}
        self.state_space = Box(-np.inf, np.inf, shape=[1, ])
        self.agents = self.env.agents
        self.num_agents = self.env.num_agents
        self.action_space = {agent: self.env.action_space(agent) for agent in self.agents}
        self.observation_space = {agent: self.env.observation_space(agent) for agent in self.agents}
        self.max_episode_steps = max_episode_steps
        self.individual_episode_reward = {k: 0.0 for k in self.agents}
        self._episode_step = 0


    def close(self):
        """Close the environment."""
        self.env.close()

    def render(self, *args):
        """Get the rendered images of the environment."""
        return self.env.render(*args)

    def reset(self):
        """Reset the environment to its initial state."""
        observations = self.env.reset()
        for agent_key in self.agents:
            self.individual_episode_reward[agent_key] = 0.0
        reset_info = {"individual_episode_rewards": self.individual_episode_reward}
        self._episode_step = 0
        return observations, reset_info

    def step(self, actions): #
        """Take an action as input, perform a step in the underlying pettingzoo environment."""
        for k, v in actions.items():
            actions[k] = np.clip(v, self.action_space[k].low, self.action_space[k].high)
        observations, rewards, terminated, truncated = self.env.step(actions)
        for k, v in rewards.items():
            self.individual_episode_reward[k] += v
        step_info = {"individual_episode_rewards": self.individual_episode_reward}
        self._episode_step += 1
        truncated = True if self._episode_step >= self.max_episode_steps else False
        return observations, rewards, terminated, truncated, step_info

    def state(self):
        """Returns the global state of the environment."""
        return self.state_space.sample()

    def agent_mask(self):
        """
        Create a boolean mask indicating which agents are currently alive.
        Note: For MPE environment, all agents are alive before the episode is terminated.
        """
        return {agent: True for agent in self.agents}

    def avail_actions(self): #
        """Returns a boolean mask indicating which actions are available for each agent."""
        return None

if __name__ == "__main__":
    REGISTRY_MULTI_AGENT_ENV[configs.env_name] = UnityMultiAgentEnv # configs.env_name: "MyNewMultiAgentEnv"
    envs = make_envs(configs)  # Make parallel environments.
    Agent = QMIX_Agents(config=configs, envs=envs)  # Create a DDPG agent from XuanCe.
    Agent.train(configs.running_steps // configs.parallels)  # Train the model for numerous steps.
    Agent.save_model("final_train_model.pth")  # Save the model to model_dir.
    Agent.finish()  # Finish the training.