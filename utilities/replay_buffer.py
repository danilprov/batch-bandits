import numpy as np


class ReplayBuffer:
    def __init__(self, size, seed):
        """
        Args:
            size (integer): The size of the replay buffer.
            minibatch_size (integer): The sample size.
            seed (integer): The seed for the random number generator.
        """
        self.buffer = []
        self.rand_generator = np.random.RandomState(seed)
        self.max_size = size

    def append(self, state, action, reward):
        """
        Args:
            state (Numpy array): The state.
            action (integer): The action.
            reward (float): The reward.
            terminal (integer): 1 if the next state is a terminal state and 0 otherwise.
            next_state (Numpy array): The next state.
        """
        if len(self.buffer) == self.max_size:
            del self.buffer[0]
        self.buffer.append([state, action, reward])

    def sample(self, last_action):
        """
        Returns:
            A list of transition tuples including state, action, reward, terinal, and next_state
        """
        state, action, reward = map(list, zip(*self.buffer))
        idxs = [elem == last_action for elem in action]
        X = [b for a, b in zip(idxs, state) if a]
        y = [b for a, b in zip(idxs, reward) if a]

        return X, y

    def size(self):
        return len(self.buffer)


if __name__ == "__main__":

    buffer = ReplayBuffer(size=100000, seed=1)
    buffer.append([1, 2, 3], 0, 1)
    buffer.append([4, 21, 3], 1, 1)
    buffer.append([0, 1, 1], 0, 0)

    print(buffer.sample(0))
