import glob
from collections import defaultdict
from typing import Tuple, Dict

import numpy as np
import pandas as pd


def count_mistakes(negatives) -> Tuple[Dict, pd.DataFrame]:
    mistakes = defaultdict(list)
    dataframes = []

    for f in negatives:
        df = pd.read_csv(f)
        for i, row in df.iterrows():
            image_id = row['image_id']
            mistake_code = str(row['y_true']) + str(row['y_pred'])
            mistakes[image_id].append(mistake_code)
            # row['mistake_code'] = mistake_code

        dataframes.append(df)

    all_mistakes = pd.concat(dataframes)
    return mistakes, all_mistakes


def main():
    # train_negatives = glob.glob('C:/Develop/Kaggle/Kaggle-2019-Blindness-Detection/negative-mining/negatives_fold0/train/*.csv') + \
    #                   glob.glob('C:/Develop/Kaggle/Kaggle-2019-Blindness-Detection/negative-mining/negatives_fold1/train/*.csv')

    valid_negatives = glob.glob(
        'D:\Develop\Kaggle\Kaggle-2019-Blindness-Detection\\negatives\\negatives\\valid\\' + '*.csv') + \
                      glob.glob(
                          'D:\Develop\Kaggle\Kaggle-2019-Blindness-Detection\\negatives\\negatives1\\valid\\' + '*.csv') + \
                      glob.glob(
                          'D:\Develop\Kaggle\Kaggle-2019-Blindness-Detection\\negatives\\negatives2\\valid\\' + '*.csv') + \
                      glob.glob(
                          'D:\Develop\Kaggle\Kaggle-2019-Blindness-Detection\\negatives\\negatives3\\valid\\' + '*.csv')

    print(len(valid_negatives))
    mistakes_per_sample, all_mistakes = count_mistakes(valid_negatives)

    all_mistakes['distance'] = np.sqrt((all_mistakes['y_true'].values - all_mistakes['y_pred'].values) ** 2)
    all_mistakes = all_mistakes.sort_values(by='distance', ascending=False)
    all_mistakes.to_csv('all_mistakes.csv', index=None)
    print(all_mistakes.head())
    grouped_by_id = all_mistakes.groupby(['image_id'])['distance'].sum().reset_index()
    grouped_by_id = grouped_by_id.sort_values(by='distance', ascending=False)

    grouped_by_id.to_csv('grouped_by_id.csv', index=None)
    print(grouped_by_id.head())

    # df1 = all_mistakes.groupby(['image_id']).size().sort_values(ascending=False).reset_index(name='count')
    # # ax = df1.hist(by='count')
    # # plt.show()
    # mean = np.mean(df1['count'])
    # median = np.median(df1['count'])
    # std = np.std(df1['count'])
    # print(mean, median, std)
    #
    # plt.figure()
    # plt.hist(df1['count'])
    # plt.show()
    #
    # df1 = df1[df1['count'] > 45]
    #
    # print(df1)
    #
    # noisy_labels = np.unique(df1['image_id'])
    # print('Noisy labels', len(noisy_labels))
    # print(noisy_labels)


if __name__ == '__main__':
    main()
