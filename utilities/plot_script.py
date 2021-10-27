import numpy as np
import matplotlib.pyplot as plt


def get_leveled_data(arr):
    """

    Args:
        arr: list of lists os different length

    Returns:
        average result over arr, axis=0

    """
    b = np.zeros([len(arr), len(max(arr, key=lambda x: len(x)))])
    b[:, :] = np.nan
    for i, j in enumerate(arr):
        b[i][0:len(j)] = j

    return b

def smooth(data, k):
    num_episodes = data.shape[1]
    num_runs = data.shape[0]

    smoothed_data = np.zeros((num_runs, num_episodes))

    for i in range(num_episodes):
        if i < k:
            smoothed_data[:, i] = np.mean(data[:, :i + 1], axis=1)
        else:
            smoothed_data[:, i] = np.mean(data[:, i - k:i + 1], axis=1)

    return smoothed_data

def plot_result(result_batch, result_online, batch_size):
    plt_agent_sweeps = []
    num_steps = np.inf

    fig, ax = plt.subplots(figsize=(8, 6))

    for data, label in zip([result_batch, result_online], ['batch', 'online']):
        sum_reward_data = get_leveled_data(data)

        # smooth data
        smoothed_sum_reward = smooth(data=sum_reward_data, k=100)

        mean_smoothed_sum_reward = np.mean(smoothed_sum_reward, axis=0)

        if mean_smoothed_sum_reward.shape[0] < num_steps:
            num_steps = mean_smoothed_sum_reward.shape[0]

        plot_x_range = np.arange(0, mean_smoothed_sum_reward.shape[0])
        graph_current_agent_sum_reward, = ax.plot(plot_x_range, mean_smoothed_sum_reward[:],
                                                  label=label)
        plt_agent_sweeps.append(graph_current_agent_sum_reward)


    update_points = np.ceil(np.arange(num_steps) / batch_size).astype(int)
    ax.plot(plot_x_range, mean_smoothed_sum_reward[update_points], label='upper bound')

    ax.legend(handles=plt_agent_sweeps, fontsize=13)
    ax.set_title("Learning Curve", fontsize=15)
    ax.set_xlabel('Episodes', fontsize=14)
    ax.set_ylabel('reward', rotation=0, labelpad=40, fontsize=14)
    # ax.set_ylim([-300, 300])

    plt.tight_layout()
    plt.show()
