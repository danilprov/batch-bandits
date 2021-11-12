import pickle
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from utilities.data_generator import generate_samples


def data_randomizer(pickle_file, seed=None):
    if isinstance(pickle_file, str):
        with open(pickle_file, 'rb') as f:
            dataset = pickle.load(f)
    else:
        dataset = pickle_file

    actions = sorted(dataset.iloc[:, -3].unique().tolist())
    tst_smpl = pd.DataFrame().reindex_like(dataset).dropna()
    ratio = 0.1

    for action in actions:
        action_subsample = dataset[dataset.iloc[:, -3] == action]
        action_drop, action_use = train_test_split(action_subsample.index, test_size=ratio,
                                                   random_state=seed,
                                                   stratify=action_subsample.iloc[:, -2])
        tst_smpl = pd.concat([tst_smpl,
                              action_subsample.loc[action_use]]).sample(frac=1, random_state=seed)

    tst_smpl = tst_smpl.reset_index(drop=True)

    del action_drop, action_use

    X = tst_smpl.iloc[:, :-3].to_numpy()
    A = tst_smpl.iloc[:, -3].to_numpy()
    Y = tst_smpl.iloc[:, -2].to_numpy()
    probs = tst_smpl.iloc[:, -1].to_numpy()

    return X, A, Y/probs


class BanditDataset(Dataset):
    def __init__(self, pickle_file, seed=None):
        # load dataset
        X, A, Y = data_randomizer(pickle_file, seed)
        self.features = X
        self.actions = A
        self.rewards = Y

    def __len__(self):
        return len(self.rewards)

    def __getitem__(self, idx):
        feature_vec = self.features[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]

        return feature_vec, action, reward


if __name__ == '__main__':
    dir = 'C:/Users/provo501/Documents/assignment/data/preprocessed_hidden_data.pickle'
    data = data_randomizer(dir)

    dataset = BanditDataset(pickle_file=dir, seed=1)
    print(len(dataset))
    print(dataset.__len__())
    print(dataset[420])
    print(dataset[421])
    print(dataset[0])
    print(dataset[1])

    dl = DataLoader(dataset, batch_size=2, shuffle=True)

    print(next(iter(dl)))

    dataset = generate_samples(100000, 4, 3, True)
    dataset = BanditDataset(pickle_file=dataset, seed=1)
    print(len(dataset))
    print(dataset.__len__())
    print(dataset[420])
    print(dataset[421])
    print(dataset[0])
    print(dataset[1])

