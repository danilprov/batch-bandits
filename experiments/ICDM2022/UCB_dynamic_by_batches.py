import numpy as np
import pickle
import os

from MAB.bandit_agents import UCBAgent
from MAB.k_arm_env import Environment
from MAB.wrapper import BanditWrapper


model_dir = './results/UCB/dynamic_by_batches'
if not os.path.exists(model_dir):
    print(f'Creating a new model directory: {model_dir}')
    os.makedirs(model_dir)

num_runs = 100
num_steps = 10001
seed = None
exper_info = {"num_runs": num_runs,
              "num_steps": num_steps,
              "seed": seed,
              "return_type": "full_regret"}

eps = [0.1, 0.02]
Ks = [2, 5, 10]

environments = []
for k in Ks:
    for e in eps:
        env = [0.5] + [0.5 - e] * (k-1)
        environments.append(env)

for arms_values in environments:
    print('run experiment for ' + str(arms_values))
    k = len(arms_values)
    reward_type = 'Bernoulli'
    env_info = {"num_actions": k,
                "reward_type": reward_type,
                "arms_values": arms_values}
    env = Environment
    agent = UCBAgent
    alpha = 1

    # run batch agent
    batches = np.logspace(1.0, 3.0, num=20).astype(int)
    batches = np.insert(batches, 0, 1)
    results = {}

    for batch in batches:
        agent_info_batch = {"num_actions": k, "batch_size": batch, "alpha": alpha}
        experiment = BanditWrapper(env, agent)
        batch_regret = experiment.get_average_performance(agent_info_batch, env_info, exper_info)
        results[batch] = batch_regret

    # save data
    e = round(arms_values[0] - arms_values[1], 2)
    name = 'dyn_by_batch_' + str(k) + '_' + str(e)
    with open(model_dir + '/' + name + '.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("End!")
