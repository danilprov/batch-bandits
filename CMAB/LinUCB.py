import numpy as np
from numpy.linalg import inv

from basics.base_agent import BaseAgent


class LinUCBAgent(BaseAgent):

    def __init__(self):
        super().__init__()
        self.name = "LinUCB"

    def agent_init(self, agent_info=None):

        if agent_info is None:
            agent_info = {}

        self.num_actions = agent_info.get('num_actions', 3)
        self.alpha = agent_info.get('alpha', 1)
        self.batch_size = agent_info.get('batch_size', 1)
        # Set random seed for policy for each run
        self.policy_rand_generator = np.random.RandomState(agent_info.get("seed", None))

        self.last_action = None
        self.last_state = None
        self.num_round = None

    def agent_policy(self, observation):
        p_t = np.zeros(self.num_actions)

        for i in range(self.num_actions):
            # initialize theta hat
            self.theta = inv(self.A[i]).dot(self.b[i])
            # get context of each arm from flattened vector of length 100
            cntx = observation
            # get gain reward of each arm
            p_t[i] = self.theta.T.dot(cntx) + self.alpha * np.sqrt(np.maximum(cntx.dot(inv(self.A[i]).dot(cntx)), 0))
        # action = np.random.choice(np.where(p_t == max(p_t))[0])
        action = self.policy_rand_generator.choice(np.where(p_t == max(p_t))[0])

        return action

    def agent_start(self, observation):
        # Specify feature dimension
        self.ndims = len(observation)

        self.A = np.zeros((self.num_actions, self.ndims, self.ndims))
        # Instantiate b as a 0 vector of length ndims.
        self.b = np.zeros((self.num_actions, self.ndims, 1))
        # set each A per arm as identity matrix of size ndims
        for arm in range(self.num_actions):
            self.A[arm] = np.eye(self.ndims)

        self.A_oracle = self.A.copy()
        self.b_oracle = self.b.copy()

        self.last_state = observation
        self.last_action = self.agent_policy(self.last_state)
        self.num_round = 0

        return self.last_action

    def agent_update(self, reward):
        self.A_oracle[self.last_action] = self.A_oracle[self.last_action] + np.outer(self.last_state, self.last_state)
        self.b_oracle[self.last_action] = np.add(self.b_oracle[self.last_action].T, self.last_state * reward).reshape(self.ndims, 1)

    def agent_step(self, reward, observation):
        if reward is not None:
            self.agent_update(reward)
            # it is a good question whether I should increment num_round outside
            # condition or not (since theoretical result doesn't clarify this
            self.num_round += 1

        if self.num_round % self.batch_size == 0:
            self.A = self.A_oracle.copy()
            self.b = self.b_oracle.copy()

        self.last_state = observation
        self.last_action = self.agent_policy(self.last_state)

        return self.last_action

    def agent_end(self, reward):
        if reward is not None:
            self.agent_update(reward)
            self.num_round += 1

        if self.num_round % self.batch_size == 0:
            self.A = self.A_oracle.copy()
            self.b = self.b_oracle.copy()

    def agent_message(self, message):
        pass

    def agent_cleanup(self):
        pass


if __name__ == '__main__':
    agent_info = {'alpha': 2,
                  'num_actions': 4,
                  'seed': 1}

    # check initialization
    linucb = LinUCBAgent()
    linucb.agent_init(agent_info)
    print(linucb.num_actions, linucb.alpha)

    assert linucb.num_actions == 4
    assert linucb.alpha == 2

    # check policy
    observation = np.array([1, 2, 5, 0])
    linucb.A = np.zeros((linucb.num_actions, len(observation), len(observation)))
    # Instantiate b as a 0 vector of length ndims.
    linucb.b = np.zeros((linucb.num_actions, len(observation), 1))
    # set each A per arm as identity matrix of size ndims
    for arm in range(linucb.num_actions):
        linucb.A[arm] = np.eye(len(observation))

    action = linucb.agent_policy(observation)
    print(action)

    assert action == 1

    # check start
    observation = np.array([1, 2, 5, 0])
    linucb.agent_start(observation)
    print(linucb.ndims)
    print(linucb.last_state, linucb.last_action)

    assert linucb.ndims == len(observation)
    assert np.allclose(linucb.last_state, observation)
    assert np.allclose(linucb.b, np.zeros((linucb.num_actions, len(observation), 1)))
    assert np.allclose(linucb.A, np.array([np.eye(len(observation)), np.eye(len(observation)),
                                           np.eye(len(observation)), np.eye(len(observation))]))
    assert linucb.last_action == 3

    # check step
    observation = np.array([5, 3, 1, 2])
    reward = 1

    action = linucb.agent_step(reward, observation)
    print(linucb.A)
    print(linucb.b)
    print(action)

    true_A = np.array([[2., 2., 5., 0.],
                       [2., 5., 10., 0.],
                       [5., 10., 26., 0.],
                       [0., 0., 0., 1.]])

    true_b = np.array([[1.],
                       [2.],
                       [5.],
                       [0.]])

    for i in range(3):
        assert np.allclose(linucb.A[i], np.eye(4))
        assert np.allclose(linucb.b[i], np.zeros((linucb.num_actions, 4, 1)))
    assert np.allclose(linucb.A[3], true_A)
    assert np.allclose(linucb.b[3], true_b)
    assert linucb.last_action == 0

    observation = np.array([3, 1, 3, 5])
    reward = None

    action = linucb.agent_step(reward, observation)
    print(linucb.A)
    print(linucb.b)
    print(action)

    assert np.allclose(linucb.A[3], true_A)
    assert np.allclose(linucb.b[3], true_b)
    assert action == 0

    # check batch size
    agent_info = {'alpha': 2,
                  'num_actions': 4,
                  'seed': 1,
                  'batch_size': 2}
    linucb = LinUCBAgent()
    linucb.agent_init(agent_info)
    observation = np.array([1, 2, 5, 0])
    linucb.agent_start(observation)
    assert linucb.num_round == 0
    assert linucb.last_action == 1

    observation = np.array([5, 3, 1, 2])
    reward = 1

    action = linucb.agent_step(reward, observation)
    assert linucb.num_round == 1
    assert np.allclose(linucb.b, np.zeros((linucb.num_actions, len(observation), 1)))
    assert np.allclose(linucb.A, np.array([np.eye(len(observation)), np.eye(len(observation)),
                                           np.eye(len(observation)), np.eye(len(observation))]))

    for i in [0, 2, 3]:
        assert np.allclose(linucb.A_oracle[i], np.eye(4))
        assert np.allclose(linucb.b_oracle[i], np.zeros((linucb.num_actions, 4, 1)))
    assert np.allclose(linucb.A_oracle[1], true_A)
    assert np.allclose(linucb.b_oracle[1], true_b)

    observation = np.array([3, 1, 3, 5])
    reward = None
    action = linucb.agent_step(reward, observation)
    # sinse reward is None, nothing should happen
    assert linucb.num_round == 1
    assert np.allclose(linucb.b, np.zeros((linucb.num_actions, len(observation), 1)))
    assert np.allclose(linucb.A, np.array([np.eye(len(observation)), np.eye(len(observation)),
                                           np.eye(len(observation)), np.eye(len(observation))]))

    for i in [0, 2, 3]:
        assert np.allclose(linucb.A_oracle[i], np.eye(4))
        assert np.allclose(linucb.b_oracle[i], np.zeros((linucb.num_actions, 4, 1)))
    assert np.allclose(linucb.A_oracle[1], true_A)
    assert np.allclose(linucb.b_oracle[1], true_b)

    observation = np.array([3, 0, 2, 5])
    reward = 0
    action = linucb.agent_step(reward, observation)

    assert linucb.num_round == 2
    assert np.allclose(linucb.b, linucb.b_oracle)
    assert np.allclose(linucb.A, linucb.A_oracle)
