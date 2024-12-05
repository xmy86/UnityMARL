from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_aec_env import UnityAECEnv
import re

unity_env_path = "combat_12_3.app"  # 替换为您的 Unity 可执行文件路径
unity_env = UnityEnvironment(file_name=unity_env_path, no_graphics=False, seed=42)
env = UnityAECEnv(unity_env)
env.reset()

class AgentInfoFromEnvPerStep:
    def __init__(self, agent, env_last):
        observation, reward, termination, info = env_last
        self.new_reward = reward
        self.termination = termination
        self.observation = observation
        self.behavior_name = info["behavior_name"]
        self.group_id = int(re.search(r"team=(\d+)", agent).group(1))
        self.group_reward = info["group_reward"]

    def __str__(self):
        return (f"observation(type, shape): {type(self.observation)}, {self.observation}\n"
                f"new_reward: {self.new_reward}\n"
                f"behavior_name: {self.behavior_name}\n"
                f"group_id: {self.group_id}\n"
                f"group_reward: {self.group_reward}")

for agent in env.agent_iter():
    all_info = AgentInfoFromEnvPerStep(agent, env.last())
    print(all_info)
    action = env.action_spaces[agent].sample()
    env.step(action)

env.close()
