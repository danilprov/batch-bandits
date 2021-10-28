import numpy as np
import pickle
import os

from MAB.bandit_agents import UCBAgent
from MAB.k_arm_env import Environment
from MAB.wrapper import BanditWrapper


model_dir = '../results/UCB/dynamic_by_batches'
if not os.path.exists(model_dir):
    print(f'Creating a new model directory: {model_dir}')
    os.makedirs(model_dir)

num_runs = 500
num_steps = 10001
seed = None
exper_info = {"num_runs": num_runs,
              "num_steps": num_steps,
              "seed": seed,
              "return_type": "regret"}

environments = [[0.7, 0.5], [0.7, 0.4], [0.7, 0.1],
                [0.35, 0.18, 0.47, 0.61],
                [0.4, 0.75, 0.57, 0.49],
                [0.70, 0.50, 0.30, 0.10]]

for arms_values in environments:
    k = len(arms_values)
    reward_type = 'Bernoulli'
    env_info = {"num_actions": k,
                "reward_type": reward_type,
                "arms_values": arms_values}
    env = Environment
    agent = UCBAgent
    alpha = 1

    # run online agent
    agent_info_online = {"num_actions": k, "batch_size": 1, "alpha": alpha}
    experiment = BanditWrapper(env, agent)
    online_regret = experiment.get_average_performance(agent_info_online, env_info, exper_info)

    # run batch agent
    batches = np.logspace(1.0, 3.0, num=20).astype(int)
    actual_regret = []
    upper_bound = []

    for batch in batches:
        agent_info_batch = {"num_actions": k, "batch_size": batch, "alpha": alpha}
        experiment = BanditWrapper(env, agent)
        batch_regret = experiment.get_average_performance(agent_info_batch, env_info, exper_info)
        actual_regret.append(batch_regret[-1])
        M = int(num_steps / batch)
        upper_bound.append(online_regret[M] * batch)

    # save data
    name = 'dyn_by_batch_' + str(arms_values)
    name1 = name + ' batch_regret'
    with open(model_dir + '/' + name1 + '.pickle', 'wb') as handle:
        pickle.dump(actual_regret, handle, protocol=pickle.HIGHEST_PROTOCOL)

    name2 = name + ' online_regret'
    with open(model_dir + '/' + name2 + '.pickle', 'wb') as handle:
        pickle.dump(online_regret, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("End!")
