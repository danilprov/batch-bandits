import numpy as np
import pickle
import matplotlib.pyplot as plt

from MAB.bandit_agents import TSAgent
from MAB.k_arm_env import Environment
from MAB.wrapper import BanditWrapper

env = Environment
agent = TSAgent

num_runs = 1000
num_steps = 10000
seed = None
if_save = False
exper_info = {"num_runs": num_runs,
              "num_steps": num_steps,
              "seed": seed,
              "return_type": "regret"}

k = 2
arms_values = [0.7, 0.65]
reward_type = 'Bernoulli'
env_info = {"num_actions": k,
            "reward_type": reward_type,
            "arms_values": arms_values}

# batch-online experiment
batch_res = []
online_res = []
batch = 10
agent_info_batch = {"num_actions": k, "batch_size": batch}
agent_info_online = {"num_actions": k, "batch_size": 1}

exp1 = BanditWrapper(env, agent)
batch_res.append(exp1.get_average_performance(agent_info_batch, env_info, exper_info))
online_res.append(exp1.get_average_performance(agent_info_online, env_info, exper_info))

av_online_res = np.mean(online_res, axis=0)
av_batch_res = np.mean(batch_res, axis=0)

plt.plot(av_batch_res, label='batch')
plt.plot(av_online_res, label='online')

M = int(num_steps / batch)
update_points = np.ceil(np.arange(num_steps) / batch).astype(int)
plt.plot(av_online_res[update_points] * batch, ls='--',
         label='upper bound, batch size = 10')
plt.title('Cumulative Regret averaged over ' + str(num_runs) + ' runs')
plt.xlabel('time steps')
plt.ylabel('regret')
plt.grid(b=True, which='major', linestyle='--', alpha=0.5)
plt.minorticks_on()
plt.grid(b=True, which='minor', linestyle=':', alpha=0.2)
plt.legend()
if if_save:
    plt.savefig('results/TS  example.png', bbox_inches='tight')
plt.show()

if if_save:
    name = 'batch_result, runs=' + str(num_runs) + ', steps=' + str(num_steps)
    with open('results/' + '/' + name + '.pickle', 'wb') as handle:
        pickle.dump(batch_res, handle, protocol=pickle.HIGHEST_PROTOCOL)
