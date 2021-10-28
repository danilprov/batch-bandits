
from basics.base_environment import BaseEnvironment

import numpy as np


class Environment(BaseEnvironment):
    """Implements the environment for an RLGlue environment

    Note:
        env_init, env_start, env_step, env_cleanup, and env_message are required
        methods.
    """

    actions = [0]

    def __init__(self):
        super().__init__()
        reward = None
        observation = None
        termination = None
        self.seed = None
        self.k = None
        self.reward_type = None
        self.custom_arms = None
        self.reward_state_term = (reward, observation, termination)
        self.count = 0
        self.arms = []
        self.subopt_gaps = None

    def env_init(self, env_info=None):
        """Setup for the environment called when the experiment first starts.

        Note:
            Initialize a tuple with the reward, first state observation, boolean
            indicating if it's terminal.
        """

        if env_info is None:
            env_info = {}
        self.k = env_info.get("num_actions", 2)
        self.reward_type = env_info.get("reward_type", "subgaussian")
        self.custom_arms = env_info.get("arms_values", None)

        if self.reward_type not in ['Bernoulli', 'subgaussian']:
            raise ValueError('Unknown reward_type: ' + str(self.reward_type))

        if self.custom_arms is None:
            if self.reward_type == 'Bernoulli':
                self.arms = np.random.uniform(0, 1, self.k)
            else:
                self.arms = np.random.randn(self.k)
        else:
            self.arms = self.custom_arms
        self.subopt_gaps = np.max(self.arms) - self.arms

        local_observation = 0  # An empty NumPy array

        self.reward_state_term = (0.0, local_observation, False)

    def env_start(self):
        """The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """

        return self.reward_state_term[1]

    def env_step(self, action):
        """A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """
        if self.reward_type == 'Bernoulli':
            reward = np.random.binomial(1, self.arms[action], 1)
        else:
            reward = self.arms[action] + np.random.randn()
        obs = self.reward_state_term[1]

        self.reward_state_term = (reward, obs, False)

        return self.reward_state_term

    def env_cleanup(self):
        """Cleanup done after the environment ends"""
        pass

    def env_message(self, message):
        """A message asking the environment for information

        Args:
            message (string): the message passed to the environment

        Returns:
            string: the response (or answer) to the message
        """
        if message == "what is the current reward?":
            return "{}".format(self.reward_state_term[0])

        # else
        return "I don't know how to respond to your message"
