import numpy as np
import pickle
import matplotlib.pyplot as plt

from utilities.plot_script import smooth

models = {'TS': {}, 'UCB': {}}
name = 'dyn_by_batch_'
expirements = {'env1': [0.7, 0.5], 'env2': [0.7, 0.4], 'env3': [0.7, 0.1],
               'env4': [0.35, 0.18, 0.47, 0.61],
               'env5': [0.4, 0.75, 0.57, 0.49], 'env6': [0.7, 0.5, 0.3, 0.1], }

# load data
for model in models.keys():
    path = 'analysis/tape transform/results/' + model + '/'
    for exper in expirements:
        name = 'dynamic_by_batches/dyn_by_batch_' + str(expirements[exper])

        with open(path + name + ' batch_regret' + '.pickle', 'rb') as f:
            actual_regret = pickle.load(f)

        models[model][exper] = actual_regret

batches = np.logspace(1.0, 3.0, num=20).astype(int)
linestyles = {'TS': ':', 'UCB': '-.'}
markers = {'env1': '1', 'env2': '.', 'env3': '*',
           'env4': '1', 'env5': '.', 'env6': '*'}
colors = {'TS': ['royalblue', 'mediumslateblue', 'mediumorchid'],
          'UCB': ['chocolate', 'tomato', 'brown']}

# for model in models.keys():
#     plt.title('Regret as a function of batch size')
#     plt.xlabel('batch size (log scale)')
#     plt.ylabel('regret')
#     plt.grid(b=True, which='major', linestyle='--', alpha=0.5)
#     plt.minorticks_on()
#     plt.grid(b=True, which='minor', linestyle=':', alpha=0.2)
#     for exper in models[model].keys():
#         plt.plot(batches, models[model][exper], marker=markers[exper],
#                  linestyle=linestyles[model], label=model + ' ' + exper)
#     plt.legend()
#     #plt.ylim([0, 150])
#     plt.xscale('log')
#     pic_name = model
#     plt.savefig('analysis/tape transform/results/' + pic_name + '_envs1-6.png', bbox_inches='tight')
#     plt.show()

# for model in models.keys():
#     plt.title('Regret as a function of batch size')
#     plt.xlabel('batch size (log scale)')
#     plt.ylabel('regret')
#     plt.grid(b=True, which='major', linestyle='--', alpha=0.5)
#     plt.minorticks_on()
#     plt.grid(b=True, which='minor', linestyle=':', alpha=0.2)
#     for exper in list(models[model].keys())[:3]:
#         plt.plot(batches, models[model][exper], marker=markers[exper],
#                  linestyle=linestyles[model], label=model + ' ' + exper)
#     plt.legend()
#     #plt.ylim([0, 150])
#     plt.xscale('log')
#     pic_name = model
#     plt.savefig('analysis/tape transform/results/' + pic_name + '_envs1-3.png', bbox_inches='tight')
#     plt.show()

# for model in models.keys():
#     plt.title('Regret as a function of batch size')
#     plt.xlabel('batch size (log scale)')
#     plt.ylabel('regret')
#     plt.grid(b=True, which='major', linestyle='--', alpha=0.5)
#     plt.minorticks_on()
#     plt.grid(b=True, which='minor', linestyle=':', alpha=0.2)
#     for exper in list(models[model].keys())[3:]:
#         plt.plot(batches, models[model][exper], marker=markers[exper],
#                  linestyle=linestyles[model], label=model + ' ' + exper)
#     plt.legend()
#     #plt.ylim([0, 150])
#     plt.xscale('log')
#     pic_name = model
#     plt.savefig('analysis/tape transform/results/' + pic_name + '_envs4-6.png', bbox_inches='tight')
#     plt.show()

plt.title('Regret as a function of batch size')
plt.xlabel('batch size (log scale)')
plt.ylabel('regret')
plt.grid(b=True, which='major', linestyle='--', alpha=0.5)
plt.minorticks_on()
plt.grid(b=True, which='minor', linestyle=':', alpha=0.2)
for model in models.keys():
    i = 0
    for exper in list(models[model].keys())[3:]:
        plt.plot(batches, models[model][exper], marker=markers[exper], color=colors[model][i],
                 linestyle=linestyles[model], label=model + ' ' + exper)
        i += 1
plt.legend()
# plt.ylim([0, 250])
plt.xscale('log')
pic_name = 'TS+UCB'
plt.savefig('analysis/tape transform/results/pictures/' + pic_name + '_envs4-6.png', bbox_inches='tight')
plt.show()

plt.title('Regret as a function of batch size')
plt.xlabel('batch size (log scale)')
plt.ylabel('regret')
plt.grid(b=True, which='major', linestyle='--', alpha=0.5)
plt.minorticks_on()
plt.grid(b=True, which='minor', linestyle=':', alpha=0.2)
for model in models.keys():
    i = 0
    for exper in list(models[model].keys())[:3]:
        plt.plot(batches, models[model][exper], marker=markers[exper], color=colors[model][i],
                 linestyle=linestyles[model], label=model + ' ' + exper)
        i += 1
plt.legend()
plt.ylim([0, 160])
plt.xscale('log')
pic_name = 'TS+UCB'
plt.savefig('analysis/tape transform/results/pictures/' + pic_name + '_envs1-3.png', bbox_inches='tight')
plt.show()

############################### CONTEXTUAL
contextual_models = {'LinTS': {}, 'LinUCB': {}}
batch_sizes = np.logspace(1.0, 2.7, num=20).astype(int)

with open('analysis/tape transform/results/contextual/TS/dynamic_by_batches/sum_reward_LinTS-10.npy', 'rb') as f:
    mean_smoothed_leveled_result1 = np.load(f)

hidden_const = np.random.rand()
# load data
for model in contextual_models.keys():
    actual_regret = []
    path = 'analysis/tape transform/results/contextual/' + model[3:] + '/'
    for batch in batch_sizes:
        name = 'dynamic_by_batches/sum_reward_' + model + '-' + str(batch)

        with open(path + name + '.npy', 'rb') as f:
            batch_result = np.load(f)

            smoothed_leveled_result1 = smooth(batch_result, 100)
            mean_smoothed_leveled_result1 = np.mean(smoothed_leveled_result1, axis=0)
            mean_smoothed_leveled_result1 = mean_smoothed_leveled_result1[~np.isnan(mean_smoothed_leveled_result1)]
            actual_regret.append(mean_smoothed_leveled_result1[-1]*hidden_const)

    contextual_models[model] = actual_regret

colors = {'LinTS': 'mediumslateblue',
          'LinUCB': 'tomato'}
linestyles = {'LinTS': '--', 'LinUCB': '-.'}

plt.title('Conversion rate as a function of batch size')
plt.xlabel('batch size (log scale)')
plt.ylabel('CR')
plt.grid(b=True, which='major', linestyle='--', alpha=0.5)
plt.minorticks_on()
plt.grid(b=True, which='minor', linestyle=':', alpha=0.2)
for model in contextual_models.keys():
    plt.plot(batch_sizes, contextual_models[model], label=model, linestyle=linestyles[model], color=colors[model])
plt.legend()
# plt.ylim([0, 160])
plt.xscale('log')
#plt.yticks(, [" "]*len(contextual_models[model]))
# plt.gca().axes.get_yaxis().label.set_visible(False)
plt.gca().axes.yaxis.set_ticklabels([])
pic_name = 'LinTS+LinUCB'
plt.savefig('analysis/tape transform/results/pictures/' + pic_name + '.png', bbox_inches='tight')
plt.show()
