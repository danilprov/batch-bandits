import numpy as np

from basics.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.num_actions = None

    def agent_init(self, agent_info=None):
        if agent_info is None:
            agent_info = {}
        self.num_actions = agent_info.get('num_actions', 2)

    def agent_start(self, observation):
        pass

    def agent_step(self, reward, observation):
        pass

    def agent_end(self, reward):
        pass

    def agent_cleanup(self):
        pass

    def agent_message(self, message):
        pass

    def agent_policy(self, observation):
        return np.random.choice(self.num_actions)


if __name__ == '__main__':
    ag = RandomAgent()
    print(ag.num_actions)

    ag.agent_init()
    print(ag.num_actions)
