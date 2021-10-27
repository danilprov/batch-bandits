import numpy as np
from tqdm import tqdm
import os
import shutil

from CMAB.rl_glue import RLGlue
from utilities.plot_script import get_leveled_data


def run_experiment(environment, agent, environment_parameters, agent_parameters,
                   experiment_parameters, save_data=True, dir='contextual'):
    rl_glue = RLGlue(environment, agent)

    # save sum of reward at the end of each episode
    agent_sum_reward = []

    env_info = environment_parameters
    agent_info = agent_parameters

    # one agent setting
    for run in tqdm(range(1, experiment_parameters["num_runs"] + 1)):
        env_info["seed"] = run

        rl_glue.rl_init(agent_info, env_info)
        rl_glue.rl_episode(0)
        agent_sum_reward.append(rl_glue.average_reward)

        # for episode in tqdm(range(1, experiment_parameters["num_episodes"] + 1)):
        #     # run episode
        #     rl_glue.rl_episode(experiment_parameters["timeout"])
        #
        #     episode_reward = rl_glue.rl_agent_message("get_sum_reward")
        #     agent_sum_reward[run - 1, episode - 1] = episode_reward

    leveled_result = get_leveled_data(agent_sum_reward)
    if save_data:
        save_name = "{}-{}".format(rl_glue.agent.name, rl_glue.agent.batch_size)
        file_dir = "results/{}".format(dir)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        np.save("{}/sum_reward_{}".format(file_dir, save_name), leveled_result)
        # shutil.make_archive('results', 'zip', 'results')

    return leveled_result


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from CMAB.replay_env import ReplayEnvironment
    from CMAB.LinUCB import LinUCBAgent
    from utilities.plot_script import smooth

    num_experements = 10
    batch_size = 100
    data_dir = 'C:/Users/provo501/Documents/assignment/data/preprocessed_data.pkl'

    experiment_parameters = {"num_runs": num_experements}
    env_info = {'pickle_file': data_dir}
    agent_info = {'alpha': 2,
                  'num_actions': 3,
                  'seed': 1,
                  'batch_size': 1}

    agent = LinUCBAgent
    environment = ReplayEnvironment

    result = run_experiment(environment, agent, env_info, agent_info, experiment_parameters, save_data=False)

    smoothed_leveled_result = smooth(result, 100)
    mean_smoothed_leveled_result = np.mean(smoothed_leveled_result, axis=0)

    path = 'asserts'
    filename = 'true_sum_reward_LinUCB'
    mean_smoothed_leveled_result_true = np.load('{}/{}.npy'.format(path, filename))

    # there are only nans after 1944 element, np.close doesn't work with nans at the end
    assert np.allclose(mean_smoothed_leveled_result[:1944], mean_smoothed_leveled_result_true[:1944])

    plt.plot(mean_smoothed_leveled_result, lw=3, ls='-.', label='online policy')
    plt.show()
