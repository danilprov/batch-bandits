from basics.base_environment import BaseEnvironment
from utilities.dataloader import BanditDataset


class ReplayEnvironment(BaseEnvironment):
    dataset: BanditDataset

    def __init__(self):
        super().__init__()
        self.counter = None
        self.last_observation = None

    def env_init(self, env_info=None):
        """
        Set parameters needed to setup the replay SavePilot environment.
        Assume env_info dict contains:
        {
            pickle_file: data directory [str]
        }
        Args:
            env_info (dict):
        """
        if env_info is None:
            env_info = {}

        directory = env_info['pickle_file']
        seed = env_info.get('seed', None)
        self.dataset = BanditDataset(directory, seed)
        self.idxs = range(self.dataset.__len__())
        self.counter = 0

    def _get_observation(self):
        idx = self.idxs[self.counter]

        return self.dataset.__getitem__(idx)

    def env_start(self):
        self.last_observation = self._get_observation()

        state = self.last_observation[0]
        reward = None
        is_terminal = False

        self.reward_state_term = (reward, state, is_terminal)
        self.counter += 1

        # return first state from the environment
        return self.reward_state_term[1]

    def env_step(self, action):
        true_action = self.last_observation[1]
        reward = self.last_observation[2]

        if true_action != action:
            reward = None

        observation = self._get_observation()
        state = observation[0]

        if self.counter == self.dataset.__len__() - 1:
            is_terminal = True
        else:
            is_terminal = False

        self.reward_state_term = (reward, state, is_terminal)

        self.last_observation = observation
        self.counter += 1

        return self.reward_state_term

    def env_cleanup(self):
        pass

    def env_message(self, message):
        pass


if __name__ == '__main__':
    data_dir = 'C:/Users/provo501/Documents/assignment/data/preprocessed_hidden_data.pickle'
    env_info = {'pickle_file': data_dir,
                'seed': 1}

    env = ReplayEnvironment()
    env.env_init(env_info)
    print(env.dataset.__len__())
    print('counter: ', env.counter)
    print('last observation: ', env.last_observation)

    assert env.counter == 0
    assert env.last_observation is None

    print('\n check env start')
    env.env_start()
    print('counter: ', env.counter)
    print('last observation: ', env.last_observation)
    print(env.reward_state_term)

    assert env.counter == 1
    assert env.reward_state_term[0] is None

    print('\ncheck env step 1')
    (reward, state, is_terminal) = env.env_step(0)
    print('counter: ', env.counter)
    print('last observation: ', env.last_observation)
    print(reward, state, is_terminal)

    assert env.counter == 2
    assert reward is None

    print('\ncheck env step 2')
    (reward, state, is_terminal) = env.env_step(1)
    print('counter: ', env.counter)
    print('last observation: ', env.last_observation)
    print(reward, state, is_terminal)

    assert reward == 0

    print('\ncheck env step 3')
    env = ReplayEnvironment()
    env.env_init(env_info)
    env.env_start()
    env.counter = 6000
    print('last observation (before step): ', env.last_observation)
    (reward, state, is_terminal) = env.env_step(0)

    print('counter: ', env.counter)
    print('last observation (after step): ', env.last_observation)
    print(reward, state, is_terminal)

    assert is_terminal is False

    env = ReplayEnvironment()
    env_info = {'pickle_file': None,
                'seed': 1,
                'dataset': {1, 2, 3}}
    env.env_init(env_info)

    print(env.dataset)




