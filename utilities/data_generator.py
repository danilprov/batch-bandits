import numpy as np
import pandas as pd

from utilities.softmax import softmax


def generate_samples(num_samples, num_features, num_arms, return_dataframe=False):
    np.random.seed(1)
    # generate pseudo features X and "true" arms' weights
    X = np.random.randint(0, 4, size=(num_samples, num_features))
    actions_weights = np.random.normal(loc=-1., scale=1, size=(num_arms, num_features))

    # apply data generating policy
    policy_weights = np.random.normal(size=(num_arms, num_features))
    action_scores = np.dot(X, policy_weights.T)
    action_probs = softmax(action_scores, tau=10)
    A = np.zeros((num_samples, 1))
    for i in range(num_samples):
        A[i, 0] = np.random.choice(range(num_arms), 1, p=action_probs[i, :])

    # store probabilities of choosing a particular action
    _rows = np.zeros_like(A, dtype=np.intp)
    _columns = A.astype(int)
    probs = action_probs[_rows, _columns]

    # calculate "true" outcomes Y
    ## broadcasting chosen actions to action weights
    matrix_multiplicator = actions_weights[_columns].squeeze()  # (num_samples x num_features) matrix
    rewards = np.sum(X * matrix_multiplicator, axis=1).reshape(-1, 1)
    Y = (np.sign(rewards) + 1) / 2

    if return_dataframe:
        column_names = ['X_' + str(i+1) for i in range(num_features)]
        X = pd.DataFrame(X, columns=column_names)
        A = pd.DataFrame(A, columns=['a'])
        Y = pd.DataFrame(Y, columns=['y'])
        probs = pd.DataFrame(probs, columns=['probs'])

        return pd.concat([X, A, Y, probs], axis=1)
    else:
        return X, A, Y, probs


dataset = generate_samples(100000, 4, 3, True)
dataset.head()
