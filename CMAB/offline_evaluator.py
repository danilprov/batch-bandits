from matplotlib import pyplot as plt
from torch.utils.data import Dataset

from basics.base_agent import BaseAgent


class OfflineEvaluator:
    def __init__(self, eval_info=None):

        if eval_info is None:
            eval_info = {}

        self.dataset = eval_info['dataset']
        self.agent = eval_info['agent']

        if not isinstance(self.dataset, Dataset):
            raise TypeError('dataset ' + "must be a " + str(Dataset))
        if not isinstance(self.agent, BaseAgent):
            raise TypeError('agent ' + "must be a " + str(BaseAgent))

        self.total_reward = None
        self.average_reward = None
        self.num_matches = None
        self.idxs = range(self.dataset.__len__())
        self.counter = None

    def eval_start(self):
        self.total_reward = 0
        self.average_reward = [0]
        self.num_matches = 0
        self.idxs = range(self.dataset.__len__())
        self.counter = 0

    def _get_observation(self):
        idx = self.idxs[self.counter]
        self.counter += 1

        return self.dataset.__getitem__(idx)

    def eval_step(self):
        observation = self._get_observation()

        state = observation[0]
        true_action = observation[1]
        reward = observation[2]

        pred_action = self.agent.agent_policy(state)

        if true_action != pred_action:
            return

        self.num_matches += 1
        aw_reward = self.average_reward[-1] + (reward - self.average_reward[-1]) / self.num_matches
        self.average_reward.append(aw_reward)
        self.total_reward += reward

    def eval_run(self):
        self.eval_start()

        while self.counter < self.dataset.__len__():
            self.eval_step()

        return self.average_reward


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from basics.random_agent import RandomAgent
    from utilities.dataloader import BanditDataset

    dir1 = 'C:/Users/provo501/Documents/GitHub/batch-bandits/experiments/data/mushroom_data_final.pickle'

    ra = RandomAgent()
    agent_info = {'num_actions': 2}
    ra.agent_init(agent_info)

    result = []
    result1 = []

    for seed_ in [1, 5, 10]:  # , 2, 3, 32, 123, 76, 987, 2134]:
        dataset = BanditDataset(pickle_file=dir1, seed=seed_)

        eval_info = {'dataset': dataset, 'agent': ra}
        evaluator = OfflineEvaluator(eval_info)

        reward = evaluator.eval_run()

        result.append(reward)
        result1.append(evaluator.total_reward)

    for elem in result:
        plt.plot(elem)
    plt.legend()
    plt.show()
