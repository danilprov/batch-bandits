import matplotlib.pyplot as plt
import numpy as np

from CMAB.replay_env import ReplayEnvironment
from CMAB.LinTS import LinTSAgent
from utilities.plot_script import smooth
from utilities.run_experiment import run_experiment

num_experiments = 10
batch_size = 100
data_dir = 'C:/Users/provo501/Documents/assignment/data/preprocessed_hidden_data.pickle'
env_info = {'pickle_file': data_dir}
output_dir = 'LinTS/dynamic_by_timesteps'

agent_info = {'alpha': 1,
              'num_actions': 3,
              'seed': 1,
              'batch_size': 1,
              'replay_buffer_size': 100000}
agent_info_batch = {'alpha': 1,
                    'num_actions': 3,
                    'seed': 1,
                    'batch_size': batch_size,
                    'replay_buffer_size': 100000}
experiment_parameters = {"num_runs": num_experiments}

agent = LinTSAgent
environment = ReplayEnvironment

online_result = run_experiment(environment, agent, env_info, agent_info,
                               experiment_parameters, True, output_dir)
batch_result = run_experiment(environment, agent, env_info, agent_info_batch,
                              experiment_parameters, True, output_dir)

smoothed_leveled_result = smooth(online_result, 100)
smoothed_leveled_result1 = smooth(batch_result, 100)

mean_smoothed_leveled_result = np.mean(smoothed_leveled_result, axis=0)
mean_smoothed_leveled_result1 = np.mean(smoothed_leveled_result1, axis=0)

num_steps = np.minimum(len(mean_smoothed_leveled_result), len(mean_smoothed_leveled_result1))
update_points = np.ceil(np.arange(num_steps) / batch_size).astype(int)

pic_filename = "results/{}/TS_transform_timesteps.png".format(output_dir)
plt.plot(mean_smoothed_leveled_result1, lw=3, label='batch, batch size = ' + str(batch_size))
plt.plot(mean_smoothed_leveled_result, lw=3, ls='-.', label='online policy')
plt.plot(mean_smoothed_leveled_result[update_points], lw=3, ls='-.', label='dumb policy')
plt.legend()
plt.xlabel('time steps')
plt.title("Smooth Cumulative Reward averaged over {} runs".format(num_experiments))
plt.ylabel('smoothed reward')
plt.grid(b=True, which='major', linestyle='--', alpha=0.5)
plt.minorticks_on()
plt.grid(b=True, which='minor', linestyle=':', alpha=0.2)
plt.savefig(pic_filename, bbox_inches='tight')
plt.show()
