from abc import abstractmethod
import numpy as np
from tqdm import tqdm
from MAB.rl_glue import RLGlue


class Policy:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.rl_glue = None

    @abstractmethod
    def get_average_performance(self, agent_info=None, env_info=None, exper_info=None):
        raise NotImplementedError


class BanditWrapper(Policy):
    def get_average_performance(self, agent_info=None, env_info=None, exper_info=None):

        if exper_info is None:
            exper_info = {}
        if env_info is None:
            env_info = {}
        if agent_info is None:
            agent_info = {}

        num_runs = exper_info.get("num_runs", 100)
        num_steps = exper_info.get("num_steps", 1000)
        return_type = exper_info.get("return_type", None)
        seed = exper_info.get("seed", None)

        np.random.seed(seed)
        seeds = np.random.randint(0, num_runs * 100, num_runs)

        all_averages = []
        subopt_arm_average = []
        best_arm = []
        worst_arm = []
        all_chosen_arm = []
        average_regret = []

        for run in tqdm(range(num_runs)):
            np.random.seed(seeds[run])

            self.rl_glue = RLGlue(self.env, self.agent)
            self.rl_glue.rl_init(agent_info, env_info)
            (first_state, first_action) = self.rl_glue.rl_start()

            worst_position = np.argmin(self.rl_glue.environment.arms)
            best_value = np.max(self.rl_glue.environment.arms)
            worst_value = np.min(self.rl_glue.environment.arms)
            best_arm.append(best_value)
            worst_arm.append(worst_value)

            scores = [0]
            averages = []
            subopt_arm = []
            chosen_arm_log = []

            cum_regret = [0]
            delta = self.rl_glue.environment.subopt_gaps[first_action]
            cum_regret.append(cum_regret[-1] + delta)

            # first action was made in rl_start, that's why run over num_steps-1
            for i in range(num_steps-1):
                reward, _, action, _ = self.rl_glue.rl_step()
                chosen_arm_log.append(action)
                scores.append(scores[-1] + reward)
                averages.append(scores[-1] / (i + 1))
                subopt_arm.append(self.rl_glue.agent.arm_count[worst_position])

                delta = self.rl_glue.environment.subopt_gaps[action]
                cum_regret.append(cum_regret[-1] + delta)

            all_averages.append(averages)
            subopt_arm_average.append(subopt_arm)
            all_chosen_arm.append(chosen_arm_log)

            average_regret.append(cum_regret)

        if return_type is None:
            returns = (np.mean(all_averages, axis=0),
                       np.mean(best_arm))
        elif return_type == 'regret':
            returns = np.mean(average_regret, axis=0)
        elif return_type == 'regret_reward':
            returns = (np.mean(average_regret, axis=0),
                       np.mean(all_averages, axis=0))
        elif return_type == 'arm_choice_analysis':
            returns = (np.mean(all_averages, axis=0),
                       np.mean(best_arm),
                       np.mean(all_chosen_arm, axis=0))
        elif return_type == 'complex':
            returns = (np.mean(all_averages, axis=0),
                       np.mean(subopt_arm_average, axis=0),
                       np.array(best_arm), np.array(worst_arm),
                       np.mean(average_regret, axis=0))

        return returns
