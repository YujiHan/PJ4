import numpy as np
import pandas as pd
import loaddata_local

output_path_local = '/home/hanyuji/projects/PJ4/data/local_features/'
data_path = '/home/hanyuji/projects/PJ4/data/food_101/meta/'

local_types = ['sift', 'orb', 'surf']
dataset_types = ['train', 'test']

dataset_dic = {}

for local_type in local_types:
    for dataset_type in dataset_types:
        print(local_type)
        list_path = data_path + dataset_type + '.txt'
        data, labels = loaddata_local.load_dataset(
            list_path, local_feature_type=local_type
        )
        np.save(
            f'{output_path_local}local_feature_{local_type}_{dataset_type}.npy', data
        )
        np.save(
            f'{output_path_local}local_label_{local_type}_{dataset_type}.npy', labels
        )

        dataset_dic[f'local_feature_{local_type}_{dataset_type}'] = data
        dataset_dic[f'local_label_{local_type}_{dataset_type}'] = labels
