import numpy as np
from scipy.stats import beta
from basics.base_agent import BaseAgent


class Agent(BaseAgent):
    """agent does *no* learning, selects random action always"""

    def __init__(self):
        super().__init__()
        self.arm_count = None
        self.last_action = None
        self.num_actions = None
        self.q_values = None
        self.step_size = None
        self.initial_value = 0.0
        self.batch_size = None
        self.q_values_oracle = None  # used for batch updates

    def agent_init(self, agent_info=None):
        """Setup for the agent called when the experiment first starts."""

        if agent_info is None:
            agent_info = {}

        self.num_actions = agent_info.get("num_actions", 2)
        self.initial_value = agent_info.get("initial_value", 0.0)
        self.q_values = np.ones(agent_info.get("num_actions", 2)) * self.initial_value
        self.step_size = agent_info.get("step_size", 0.1)
        self.batch_size = agent_info.get('batch_size', 1)
        self.q_values_oracle = self.q_values.copy()
        self.arm_count = np.zeros(self.num_actions)  # [0.0 for _ in range(self.num_actions)]
        # self.last_action = np.random.choice(self.num_actions)  # set first action to random

    def agent_start(self, observation):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            observation (Numpy array): the state observation from the
                environment's evn_start function.
        Returns:
            The first action the agent takes.
        """
        self.last_action = np.random.choice(self.num_actions)

        return self.last_action

    def agent_step(self, reward, observation):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            observation (Numpy array): the state observation from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """
        # local_action = 0  # choose the action here
        self.last_action = np.random.choice(self.num_actions)

        return self.last_action

    def agent_end(self, reward):
        pass

    def agent_cleanup(self):
        pass

    def agent_message(self, message):
        pass


def argmax(q_values):
    """
    Takes in a list of q_values and returns the index of the item
    with the highest value. Breaks ties randomly.
    returns: int - the index of the highest value in q_values
    """
    top_value = float("-inf")
    ties = []

    for i in range(len(q_values)):
        if q_values[i] > top_value:
            ties = [i]
            top_value = q_values[i]
        elif q_values[i] == top_value:
            ties.append(i)

    return np.random.choice(ties)


class GreedyAgent(Agent):
    def __init__(self):
        super().__init__()

    def agent_init(self, agent_info=None):
        if agent_info is None:
            agent_info = {}

        super().agent_init(agent_info)

    def agent_step(self, reward, observation):
        """
        Takes one step for the agent. It takes in a reward and observation and
        returns the action the agent chooses at that time step.

        Arguments:
        reward -- float, the reward the agent received from the environment after taking the last action.
        observation -- float, the observed state the agent is in. Do not worry about this as you will not use it
                              until future lessons
        Returns:
        current_action -- int, the action chosen by the agent at the current time step.
        """

        a = self.last_action
        self.arm_count[a] += 1
        self.q_values_oracle[a] = self.q_values_oracle[a] + 1 / self.arm_count[a] * (reward - self.q_values_oracle[a])

        if sum(self.arm_count) % self.batch_size == 0:
            self.q_values = self.q_values_oracle.copy()

        current_action = argmax(self.q_values)
        self.last_action = current_action

        return current_action


class EpsilonGreedyAgent(Agent):
    def __init__(self):
        super().__init__()
        self.epsilon = None

    def agent_init(self, agent_info=None):
        if agent_info is None:
            agent_info = {}

        super().agent_init(agent_info)
        self.epsilon = agent_info.get("epsilon", 0.1)

    def agent_step(self, reward, observation):
        """
        Takes one step for the agent. It takes in a reward and observation and
        returns the action the agent chooses at that time step.

        Arguments:
        reward -- float, the reward the agent received from the environment after taking the last action.
        observation -- float, the observed state the agent is in. Do not worry about this as you will not use it
                              until future lessons
        Returns:
        current_action -- int, the action chosen by the agent at the current time step.
        """

        a = self.last_action

        self.arm_count[a] += 1
        self.q_values_oracle[a] = self.q_values_oracle[a] + 1 / self.arm_count[a] * (reward - self.q_values_oracle[a])

        if np.sum(self.arm_count) % self.batch_size == 0:
            self.q_values = self.q_values_oracle.copy()

        if np.random.random() < self.epsilon:
            current_action = np.random.choice(range(len(self.arm_count)))
        else:
            current_action = argmax(self.q_values)

        self.last_action = current_action

        return current_action


class UCBAgent(Agent):
    def __init__(self):
        super().__init__()
        self.upper_bounds = None
        self.alpha = None  # exploration parameter

    def agent_init(self, agent_info=None):
        if agent_info is None:
            agent_info = {}

        super().agent_init(agent_info)
        self.alpha = agent_info.get("alpha", 1.0)
        self.arm_count = np.ones(self.num_actions)
        self.upper_bounds = np.sqrt(np.log(np.sum(self.arm_count)) / self.arm_count)

    def agent_step(self, reward, observation):
        a = self.last_action

        self.arm_count[a] += 1
        self.q_values_oracle[a] = self.q_values_oracle[a] + 1 / self.arm_count[a] * (reward - self.q_values_oracle[a])

        # since we start with arms_count = np.ones(num_actions),
        # we should subtract num_actions to get number of the current round
        if (np.sum(self.arm_count) - self.num_actions) % self.batch_size == 0:
            self.q_values = self.q_values_oracle.copy()
            self.upper_bounds = np.sqrt(np.log(np.sum(self.arm_count)) / self.arm_count)

        # if min(self.q_values + self.alpha * self.upper_bounds) < max(self.q_values):
        #     print(f'Distinguish suboptimal arm at step {sum(self.arm_count)}')
        current_action = argmax(self.q_values + self.alpha * self.upper_bounds)
        # current_action = np.argmax(self.q_values + self.alpha * self.upper_bounds)

        self.last_action = current_action

        return current_action


class TSAgent(Agent):
    def agent_step(self, reward, observation):
        a = self.last_action
        self.arm_count[a] += 1
        self.q_values_oracle[a] = self.q_values_oracle[a] + 1 / self.arm_count[a] * (reward - self.q_values_oracle[a])

        if (np.sum(self.arm_count) - self.num_actions) % self.batch_size == 0:
            self.q_values = self.q_values_oracle.copy()

        # sample from posteriors
        theta = [beta.rvs(a + 1, b + 1, size=1) for a, b in
                 zip(self.q_values * self.arm_count, self.arm_count - self.q_values * self.arm_count)]
        # choose the max realization
        current_action = argmax(theta)
        self.last_action = current_action

        return current_action
