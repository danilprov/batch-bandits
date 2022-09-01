import numpy as np
import pickle
import matplotlib.pyplot as plt
import glob
import os
import ntpath


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

batch_sizes = np.logspace(1.0, 3.0, num=20).astype(int)

path = r'results/'
models = ['UCB']
latest_folder = 'dynamic_by_batches/'
list_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

fig1, ax1 = plt.subplots(figsize=(4,2.7))
ax1.plot(np.insert(batch_sizes, 0, 1)[:-4], np.insert(batch_sizes, 0, 1)[:-4], label='bound', linestyle='--', color='black')
for model in models:
    #all_files = glob.glob(path + model + '/' + latest_folder + "*.pickle")
    all_files = [#'dyn_by_batch_2_0.1.pickle',
                 'dyn_by_batch_2_0.02.pickle',
                 #'dyn_by_batch_5_0.1.pickle',
                 'dyn_by_batch_5_0.02.pickle',
                 #'dyn_by_batch_10_0.1.pickle',
                 'dyn_by_batch_10_0.02.pickle'
                 ]
    j = 0
    for file in all_files:
        with open(path + model + '/dynamic_by_batches/' + file, 'rb') as f:
            results = pickle.load(f)

        #alg = model + path_leaf(file)[12:-7]
        alg = ['2', '5', '10']
        markers = ['1', '.', '*']

        i = 0
        batches = list(results.keys())
        final_result = np.zeros((2, len(batches)))
        for batch in batches:
            batch_regret = results[batch]
            batch_regret = np.array([np.array(xi) for xi in batch_regret])
            mean_reward, std_reward = batch_regret.mean(axis=0), batch_regret.std(axis=0)
            nanmean_reward, nanstd_reward = np.nanmean(batch_regret, axis=0), np.nanstd(batch_regret, axis=0)

            final_result[0, i] = mean_reward[~np.isnan(mean_reward)][-1]
            final_result[1, i] = std_reward[~np.isnan(std_reward)][-1]

            #ax.plot(mean_reward, linestyle='-.', label=str(batch), lw=1)
            #ax.fill_between(range(len(mean_reward)), mean_reward - std_reward, mean_reward + std_reward, alpha=0.2)
            #ax.plot(nanmean_reward, label=str(batch), linestyle='-.', lw=1, color=list_colors[i])
            #ax.fill_between(range(len(nanmean_reward)), nanmean_reward - nanstd_reward, nanmean_reward + nanstd_reward, alpha=0.2)
            i += 1

        ax1.plot(batches, final_result[0], linestyle='-.', lw=1, label='K=' + alg[j], marker=markers[j], markevery=5)
        ax1.fill_between(batches, final_result[0] - final_result[1], final_result[0] + final_result[1], alpha=0.2)
        j += 1

# lines_labels = ax1.get_legend_handles_labels()
# lines, labels = [sum(_, []) for _ in zip(*lines_labels)]
#fig1.legend(loc='upper center')
plt.legend() #prop={'size': 3})
#plt.ylim([0.02, 0.07])
plt.xscale('log')
ax1.grid(b=True, which='major', linestyle='--', alpha=0.5)
ax1.minorticks_on()
ax1.grid(b=True, which='minor', linestyle=':', alpha=0.2)
plt.xlabel('batch size (log scale)', fontsize=10)
plt.ylabel('Regret', fontsize=10)
plt.savefig(path + model + '/dynamic_by_batches/' + 'regret_' + model  + '_0.02' + '.png', bbox_inches='tight')
plt.show()
