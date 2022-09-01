import numpy as np
import pickle
import matplotlib.pyplot as plt

from utilities.plot_script import smooth

############################### CONTEXTUAL
#contextual_models = {'LinTS': {}, 'LinUCB': {}}
contextual_models = {'LinTS': {}}
relative_contextual_model = {'LinTS': {}, 'LinUCB': {}}
batch_sizes = np.logspace(1.0, 2.7, num=20).astype(int)

hidden_const = 1 #np.random.rand()
for model in contextual_models.keys():
    actual_regret = []
    actual_std = []
    # load results for private dataset from local folder
    path = 'C:/Users/provo501/Documents/GitHub/bandits/analysis/tape transform/results/contextual/' + model[3:] + '/'
    for batch in batch_sizes:
        name = 'dynamic_by_batches/sum_reward_' + model + '-' + str(batch)

        with open(path + name + '.npy', 'rb') as f:
            batch_result = np.load(f)

            smoothed_leveled_result1 = smooth(batch_result, 100)
            mean_smoothed_leveled_result1 = np.mean(smoothed_leveled_result1, axis=0)
            mean_smoothed_leveled_result1 = mean_smoothed_leveled_result1[~np.isnan(mean_smoothed_leveled_result1)]
            std_smoothed_leveled_result1 = np.std(smoothed_leveled_result1, axis=0)
            std_smoothed_leveled_result1 = std_smoothed_leveled_result1[~np.isnan(std_smoothed_leveled_result1)]
            actual_regret.append(mean_smoothed_leveled_result1[-1]*hidden_const)
            actual_std.append(std_smoothed_leveled_result1[-1]*hidden_const)

            if batch == 10:
                idxs = np.around(len(mean_smoothed_leveled_result1) / batch_sizes).astype(int) - 1
                idxs = idxs[1:]
                idxs = np.insert(idxs, 0, len(mean_smoothed_leveled_result1) - 1)
                lower_bound = mean_smoothed_leveled_result1[idxs]

    contextual_models[model]['mean'] = actual_regret
    contextual_models[model]['std_left'] = [x - y for x, y in zip(actual_regret, actual_std)]
    contextual_models[model]['std_right'] = [x + y for x, y in zip(actual_regret, actual_std)]
    relative_contextual_model[model]['mean'] = actual_regret / actual_regret[0]
    relative_contextual_model[model]['std_left'] = [(x - y) / actual_regret[0] for x, y in zip(actual_regret, actual_std)]  # (actual_regret - actual_std) / actual_regret[0]
    relative_contextual_model[model]['std_right'] = [(x + y) / actual_regret[0] for x, y in zip(actual_regret, actual_std)]  # (actual_regret + actual_std) / actual_regret[0]

    relative_contextual_model[model]['lower_bound'] = lower_bound / actual_regret[0]

colors = {'LinTS': 'mediumslateblue',
          'LinUCB': 'tomato'}
linestyles = {'LinTS': '--', 'LinUCB': '-.'}

fig, ax = plt.subplots()
plt.title('Conversion rate as a function of batch size')
plt.xlabel('batch size (log scale)')
plt.ylabel('CR')
plt.grid(b=True, which='major', linestyle='--', alpha=0.5)
plt.minorticks_on()
plt.grid(b=True, which='minor', linestyle=':', alpha=0.2)
# ax.plot(batch_sizes, lower_bound, label='bound', linestyle='--', color='black')
for model in contextual_models.keys():
    ax.plot(batch_sizes, contextual_models[model]['mean'], label=model, linestyle=linestyles[model], color=colors[model])
    ax.fill_between(batch_sizes, contextual_models[model]['std_left'], contextual_models[model]['std_right'], alpha=0.2)
plt.legend()
plt.xscale('log')
pic_name = 'LinTS+LinUCB'
#plt.savefig('analysis/tape transform/results/pictures/' + pic_name + '.png', bbox_inches='tight')
plt.show()

colors = {'LinTS': 'mediumslateblue',
          'LinUCB': 'tomato'}
linestyles = {'LinTS': '-.', 'LinUCB': '-.'}
markers = {'LinTS': '.', 'LinUCB': '*'}

fig, ax = plt.subplots(figsize=(4,2.7))
#plt.title('Relative conversion rate as a function of batch size')
plt.xlabel('batch size (log scale)')
plt.ylabel('% of online performance')
for model in contextual_models.keys():
    ax.plot(batch_sizes, relative_contextual_model[model]['mean'], label=model, linestyle=linestyles[model], marker=markers[model])
    ax.fill_between(batch_sizes, relative_contextual_model[model]['std_left'], relative_contextual_model[model]['std_right'], alpha=0.2)
    ax.plot(batch_sizes, relative_contextual_model[model]['lower_bound'], label='bound', linestyle='--', color='black')

plt.legend()
plt.xscale('log')
plt.grid(b=True, which='major', linestyle='--', alpha=0.5)
plt.minorticks_on()
plt.grid(b=True, which='minor', linestyle=':', alpha=0.2)
plt.savefig('./results/' + model + '.png', bbox_inches='tight')
plt.show()


