import torch
import numpy as np
from argparse import Namespace
from xuance.common import Optional
from xuance.environment import DummyVecEnv
from xuance.torch.utils import NormalizeFunctions, ActivationFunctions
from xuance.torch.policies import REGISTRY_Policy
from xuance.torch.agents import OffPolicyAgent


class SACDIS_Agent(OffPolicyAgent):
    """The implementation of SAC agent with discrete actions.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
    """
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecEnv):
        super(SACDIS_Agent, self).__init__(config, envs)

        self.policy = self._build_policy()  # build policy
        self.memory = self._build_memory()
        self.learner = self._build_learner(self.config, self.policy, -np.prod(self.action_space.n).item())

    def _build_policy(self):
        normalize_fn = NormalizeFunctions[self.config.normalize] if hasattr(self.config, "normalize") else None
        initializer = torch.nn.init.orthogonal_
        activation = ActivationFunctions[self.config.activation]
        device = self.device

        # build representations.
        representation = self._build_representation(self.config.representation, self.observation_space, self.config)

        # build policy
        if self.config.policy == "Categorical_SAC":
            policy = REGISTRY_Policy["Categorical_SAC"](
                action_space=self.action_space, representation=representation,
                actor_hidden_size=self.config.actor_hidden_size, critic_hidden_size=self.config.critic_hidden_size,
                normalize=normalize_fn, initialize=initializer, activation=activation, device=device)
        else:
            raise AttributeError(f"SACDIS agent currently does not support the policy named {self.config.policy}.")

        return policy

    def action(self, observations: np.ndarray,
               test_mode: Optional[bool] = False):
        """Returns actions and values.

        Parameters:
            observations (np.ndarray): The observation.
            test_mode (Optional[bool]): True for testing without noises.

        Returns:
            actions: The actions to be executed.
            values: The evaluated values.
            dists: The policy distributions.
            log_pi: Log of stochastic actions.
        """
        _, actions_output = self.policy(observations)
        actions = actions_output.detach().cpu().numpy()
        return {"actions": actions}
