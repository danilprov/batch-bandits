import matplotlib.pyplot as plt
import numpy as np

from CMAB.replay_env import ReplayEnvironment
from CMAB.LinUCB import LinUCBAgent
from utilities.plot_script import smooth
from utilities.run_experiment import run_experiment


num_experiments = 20
data_dir = 'C:/Users/provo501/Documents/assignment/data/preprocessed_hidden_data.pickle'
env_info = {'pickle_file': data_dir}
output_dir = 'LinUCB/dynamic_by_batches'

agent_info = {'alpha': 2,
              'num_actions': 3,
              'seed': 1,
              'batch_size': 1}
experiment_parameters = {"num_runs": num_experiments}

agent = LinUCBAgent
environment = ReplayEnvironment

# run online agent
online_result = run_experiment(environment, agent, env_info, agent_info,
                               experiment_parameters, True, output_dir)
# smooth and average the result
smoothed_leveled_result = smooth(online_result, 100)
mean_smoothed_leveled_result = np.mean(smoothed_leveled_result, axis=0)
mean_smoothed_leveled_result = mean_smoothed_leveled_result[~np.isnan(mean_smoothed_leveled_result)]

# run batch agent
batch_sizes = np.logspace(1.0, 2.7, num=20).astype(int)
actual_regret = []
upper_bound = []
for batch in batch_sizes:
    agent_info_batch = {'alpha': 2,
                        'num_actions': 3,
                        'seed': 1,
                        'batch_size': batch}
    batch_result = run_experiment(environment, agent, env_info, agent_info_batch,
                                  experiment_parameters, True, output_dir)
    # smooth and average the result
    smoothed_leveled_result1 = smooth(batch_result, 100)
    mean_smoothed_leveled_result1 = np.mean(smoothed_leveled_result1, axis=0)
    mean_smoothed_leveled_result1 = mean_smoothed_leveled_result1[~np.isnan(mean_smoothed_leveled_result1)]

    actual_regret.append(mean_smoothed_leveled_result1[-1])

    # fetch dumb result
    M = int(len(mean_smoothed_leveled_result1) / batch)
    upper_bound.append(mean_smoothed_leveled_result[M])

pic_filename = "results/{}/UCB_transform_batchsize.png".format(output_dir)
plt.plot(batch_sizes, actual_regret, label='actual regret')
plt.plot(batch_sizes, [mean_smoothed_leveled_result[-1]]*len(batch_sizes), label='online policy')
plt.plot(batch_sizes, upper_bound, label='dumb policy')
plt.legend()
plt.title("Reward as a f-n of batch size (each point is averaged over {} runs)".format(num_experiments))
plt.xlabel('batch size (log scale)')
plt.ylabel('reward')
plt.grid(b=True, which='major', linestyle='--', alpha=0.5)
plt.minorticks_on()
plt.grid(b=True, which='minor', linestyle=':', alpha=0.2)
plt.savefig(pic_filename, bbox_inches='tight')
plt.show()
