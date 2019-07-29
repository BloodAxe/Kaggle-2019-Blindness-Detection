import argparse

import numpy as np
import pandas as pd
from pytorch_toolbelt.utils.fs import auto_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-df', type=str)
    parser.add_argument('-nf', '--num-folds', default=4)
    args = parser.parse_args()

    num_folds = args.num_folds

    train_set = pd.read_csv('data/train.csv')
    df = pd.read_csv(auto_file(args.df))
    train_set = train_set.merge(df, on='id_code')
    train_set = train_set.sort_values(by='is_test')

    folds = np.arange(num_folds).tolist() * len(train_set)
    folds = folds[:len(train_set)]
    train_set['fold'] = folds

    train_set.to_csv('data/train_with_folds.csv', index=None)


if __name__ == '__main__':
    main()
