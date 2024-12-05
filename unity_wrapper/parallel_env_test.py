# /site-packages/mlagents_envs/envs/unity_pettingzoo_base_env.py 
# 135 int32 -> float32

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv

unity_env_path = "combat_12_3.app"  # 替换为您的 Unity 可执行文件路径
unity_env = UnityEnvironment(file_name=unity_env_path, no_graphics=False, seed=42)
env = UnityParallelEnv(unity_env)
obsercations = env.reset() # 没有info

while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations = env.step(actions) # 没有info

env.close()