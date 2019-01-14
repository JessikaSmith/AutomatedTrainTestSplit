import pandas as pd


# header (?)
def load_data(fname, header=True):
    dataset = pd.read_csv(fname, header=header)
    dataset = dataset.sample(frac=1).drop_index(True)
    return dataset

def concatenate_dfs(path1, path2):
    test = pd.read_csv(path1)
    train = pd.read_csv(path2)
    total_dataset = pd.concat([test, train])
    total_dataset = total_dataset.sample(frac=1).reset_index(drop=True)
    return total_dataset