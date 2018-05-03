from sklearn.datasets import fetch_lfw_people

import numpy as np
import pandas as pd

TARGET_PATH = './lfw_data/'

def convert_id_to_name(idx, names_idx):
    names = np.ndarray(shape=idx.shape, dtype='<U35')
    for i in idx:
        names[i] = names[idx[i]]
    return names

lfw_bunch                  = fetch_lfw_people(data_home=TARGET_PATH)
target_names               = convert_id_to_name(lfw_bunch['target'], lfw_bunch['target_names'])
lfw_labels                 = pd.DataFrame()
lfw_labels['target_names'] = pd.Series(target_names)
lfw_labels['target_id']    = pd.Series(lfw_bunch['target'])
lfw_images                 = lfw_bunch['data']
