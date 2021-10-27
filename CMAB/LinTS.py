import numpy as np
from scipy.optimize import minimize

from basics.base_agent import BaseAgent
from utilities.replay_buffer import ReplayBuffer
from utilities.softmax import softmax


class LinTSAgent(BaseAgent):

    def __init__(self):
        super().__init__()
        self.name = "LinTS"

    def agent_init(self, agent_info=None):

        if agent_info is None:
            agent_info = {}

        self.num_actions = agent_info.get('num_actions', 3)
        self.alpha = agent_info.get('alpha', 1)
        self.lambda_ = agent_info.get('lambda', 1)
        self.batch_size = agent_info.get('batch_size', 1)
        # Set random seed for policy for each run
        self.policy_rand_generator = np.random.RandomState(agent_info.get("seed", None))

        self.replay_buffer = ReplayBuffer(agent_info['replay_buffer_size'],
                                          agent_info.get("seed"))


        self.last_action = None
        self.last_state = None
        self.num_round = None

    def agent_policy(self, observation, mode='sample'):
        p_t = np.zeros(self.num_actions)
        cntx = observation

        for i in range(self.num_actions):
            # sampling weights after update
            self.w = self.get_weights(i)

            # using weight depending on mode
            if mode == 'sample':
                w = self.w  # weights are samples of posteriors
            elif mode == 'expected':
                w = self.m[i]  # weights are expected values of posteriors
            else:
                raise Exception('mode not recognized!')

            # calculating probabilities
            p_t[i] = 1 / (1 + np.exp(-1 * cntx.dot(w)))
            action = self.policy_rand_generator.choice(np.where(p_t == max(p_t))[0])
            # probs = softmax(p_t.reshape(1, -1))
            # action = self.policy_rand_generator.choice(a=range(self.num_actions), p=probs)

        return action

    def get_weights(self, arm):
        return np.random.normal(self.m[arm], self.alpha * self.q[arm] ** (-1.0), size=len(self.w))

        # the loss function
    def loss(self, w, *args):
        X, y = args
        return 0.5 * (self.q[self.last_action] * (w - self.m[self.last_action])).dot(w - self.m[self.last_action]) + np.sum(
            [np.log(1 + np.exp(-y[j] * w.dot(X[j]))) for j in range(y.shape[0])])

    # the gradient
    def grad(self, w, *args):
        X, y = args
        return self.q[self.last_action] * (w - self.m[self.last_action]) + (-1) * np.array(
            [y[j] * X[j] / (1. + np.exp(y[j] * w.dot(X[j]))) for j in range(y.shape[0])]).sum(axis=0)

    # fitting method
    def agent_update(self, X, y):
        # step 1, find w
        self.w = minimize(self.loss, self.w, args=(X, y), jac=self.grad, method="L-BFGS-B",
                          options={'maxiter': 20, 'disp': False}).x
        # self.m_oracle[self.last_action] = self.w
        self.m[self.last_action] = self.w

        # step 2, update q
        P = (1 + np.exp(1 - X.dot(self.m[self.last_action]))) ** (-1)
        #self.q_oracle[self.last_action] = self.q[self.last_action] + (P * (1 - P)).dot(X ** 2)
        self.q[self.last_action] = self.q[self.last_action] + (P * (1 - P)).dot(X ** 2)

    def agent_start(self, observation):
        # Specify feature dimension
        self.ndims = len(observation)

        # initializing parameters of the model
        self.m = np.zeros((self.num_actions, self.ndims))
        self.q = np.ones((self.num_actions, self.ndims)) * self.lambda_
        # initializing weights using any arm (e.g. 0) because they all equal
        self.w = np.array([0.]*self.ndims, dtype=np.float64)

        # self.m_oracle = self.m.copy()
        # self.q_oracle = self.q.copy()

        self.last_state = observation
        self.last_action = self.agent_policy(self.last_state)
        self.num_round = 0

        return self.last_action


    def agent_step(self, reward, observation):
        # Append new experience to replay buffer
        if reward is not None:
            self.replay_buffer.append(self.last_state, self.last_action, reward)
            # it is a good question whether I should increment num_round outside
            # condition or not (since theoretical result doesn't clarify this
            self.num_round += 1

            if self.num_round % self.batch_size == 0:
                X, y = self.replay_buffer.sample(self.last_action)
                X = np.array(X)
                y = np.array(y)
                self.agent_update(X, y)
                # self.m = self.m_oracle.copy()
                # self.q = self.q_oracle.copy()

        self.last_state = observation
        self.last_action = self.agent_policy(self.last_state)

        return self.last_action

    def agent_end(self, reward):
        # Append new experience to replay buffer
        if reward is not None:
            self.replay_buffer.append(self.last_state, self.last_action, reward)
            # it is a good question whether I should increment num_round outside
            # condition or not (since theoretical result doesn't clarify this
            self.num_round += 1

            if self.num_round % self.batch_size == 0:
                X, y = self.replay_buffer.sample(self.last_action)
                X = np.array(X)
                y = np.array(y)
                self.agent_update(X, y)
                # self.m = self.m_oracle.copy()
                # self.q = self.q_oracle.copy()

    def agent_message(self, message):
        pass

    def agent_cleanup(self):
        pass


if __name__ == '__main__':
    agent_info = {'alpha': 2,
                  'num_actions': 3,
                  'seed': 1,
                  'lambda': 2,
                  'replay_buffer_size': 100000}

    np.random.seed(1)
    # check initialization
    lints = LinTSAgent()
    lints.agent_init(agent_info)
    print(lints.num_actions, lints.alpha, lints.lambda_)

    assert lints.num_actions == 3
    assert lints.alpha == 2
    assert lints.lambda_ == 2

    # check agent policy
    observation = np.array([1, 2, 5, 0])
    lints.m = np.zeros((lints.num_actions, len(observation)))
    lints.q = np.ones((lints.num_actions, len(observation))) * lints.lambda_
    lints.w = np.random.normal(lints.m[0], lints.alpha * lints.q[0] ** (-1.0), size=len(observation))
    print(lints.w)
    action = lints.agent_policy(observation)
    print(action)

    # check agent start
    observation = np.array([1, 2, 5, 0])
    lints.agent_start(observation)
    # manually reassign w to np.random.normal, because I np.seed doesn't work inside the class
    np.random.seed(1)
    lints.w = np.random.normal(lints.m[0], lints.alpha * lints.q[0] ** (-1.0), size=len(observation))
    print(lints.ndims)
    print(lints.last_state, lints.last_action)
    print(lints.last_action)
    assert lints.ndims == len(observation)
    assert np.allclose(lints.last_state, observation)
    assert np.allclose(lints.m, np.zeros((lints.num_actions, lints.ndims)))
    assert np.allclose(lints.q, np.ones((lints.num_actions, lints.ndims)) * lints.lambda_)
    assert np.allclose(lints.w, np.array([ 1.62434536, -0.61175641, -0.52817175, -1.07296862]))
    # assert lints.last_action == 1

    # check step
    observation = np.array([5, 3, 1, 2])
    reward = 1
    action = lints.agent_step(reward, observation)
    print(action)

    observation = np.array([1, 3, 2, 1])
    reward = 0
    action = lints.agent_step(reward, observation)
    print(action)
