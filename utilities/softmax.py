import numpy as np


def softmax(action_values, tau=1.0):
    """
    Args:
        action_values (Numpy array): A 2D array of shape (batch_size, num_actions).
                       The action-values computed by an action-value network.
        tau (float): The temperature parameter scalar.
    Returns:
        A 2D array of shape (batch_size, num_actions). Where each column is a probability distribution over
        the actions representing the policy.
    """

    # Compute the preferences by dividing the action-values by the temperature parameter tau
    preferences = action_values / tau
    # Compute the maximum preference across the actions
    max_preference = np.max(preferences, axis=1)

    # your code here

    # Reshape max_preference array which has shape [Batch,] to [Batch, 1]. This allows NumPy broadcasting
    # when subtracting the maximum preference from the preference of each action.
    reshaped_max_preference = max_preference.reshape((-1, 1))
    # print(reshaped_max_preference)

    # Compute the numerator, i.e., the exponential of the preference - the max preference.
    exp_preferences = np.exp(preferences - reshaped_max_preference)
    # print(exp_preferences)
    # Compute the denominator, i.e., the sum over the numerator along the actions axis.
    sum_of_exp_preferences = np.sum(exp_preferences, axis=1)
    # print(sum_of_exp_preferences)

    # your code here

    # Reshape sum_of_exp_preferences array which has shape [Batch,] to [Batch, 1] to  allow for NumPy broadcasting
    # when dividing the numerator by the denominator.
    reshaped_sum_of_exp_preferences = sum_of_exp_preferences.reshape((-1, 1))
    # print(reshaped_sum_of_exp_preferences)

    # Compute the action probabilities according to the equation in the previous cell.
    action_probs = exp_preferences / reshaped_sum_of_exp_preferences
    # print(action_probs)

    # your code here

    # squeeze() removes any singleton dimensions. It is used here because this function is used in the
    # agent policy when selecting an action (for which the batch dimension is 1.) As np.random.choice is used in
    # the agent policy and it expects 1D arrays, we need to remove this singleton batch dimension.
    action_probs = action_probs.squeeze()
    return action_probs


if __name__ == '__main__':
    rand_generator = np.random.RandomState(0)
    action_values = rand_generator.normal(0, 1, (2, 4))
    tau = 0.5

    action_probs = softmax(action_values, tau)
    print("action_probs", action_probs)

    assert (np.allclose(action_probs, np.array([
        [0.25849645, 0.01689625, 0.05374514, 0.67086216],
        [0.84699852, 0.00286345, 0.13520063, 0.01493741]
    ])))

    action_values = np.array([[0.0327, 0.0127, 0.0688]])
    tau = 1.
    action_probs = softmax(action_values, tau)
    print("action_probs", action_probs)

    assert np.allclose(action_probs, np.array([0.3315, 0.3249, 0.3436]), atol=1e-04)

    print("Passed the asserts! (Note: These are however limited in scope, additional testing is encouraged.)")