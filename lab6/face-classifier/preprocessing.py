from sklearn.datasets import fetch_lfw_people

import numpy as np
import pandas as pd

TARGET_PATH = './lfw_data/'

def convert_id_to_name(idx, names_idx):
    names = np.ndarray(shape=idx.shape, dtype='<U35')
    for i in idx:
        names[i] = names[idx[i]]
    return names

def load_lfw_data(path=TARGET_PATH, mfpp=0, mirror=False):
    """ Load the Labeled Faces in the Wild Datasetself.

    Args:
        path (str): The path to the directory of the cache.
        pfpp (int): The minimum training instances (of a particular label) to
            accept.
        mirror (bool): If true, will create mirrored images for every training
            instance. Note that this will double the dataset.

    Returns:
        lfw_images (ndarray): A 2d array with rows as the instances and columns
            as the flattened pixels of each image.
        lfw_labels (DataFrame): A DataFrame containing the target_id (the label)
            and the target_name (the persons name).
    """

    lfw_bunch                  = fetch_lfw_people(data_home=TARGET_PATH,
                                                  min_faces_per_person=mfpp)
    target_names               = convert_id_to_name(lfw_bunch['target'], lfw_bunch['target_names'])
    lfw_labels                 = pd.DataFrame()
    lfw_labels['target_names'] = pd.Series(target_names)
    lfw_labels['target_id']    = pd.Series(lfw_bunch['target'])
    lfw_images                 = lfw_bunch['images'] / 255
    lfw_images                 = lfw_images.reshape((-1, lfw_images.shape[1],
                                                         lfw_images.shape[2],1))

    if mirror:
        reverse_lfw_images = np.flip(lfw_images, axis=2)
        lfw_images = np.append(lfw_images, reverse_lfw_images, axis=0)
        lfw_labels = lfw_labels.append(lfw_labels)

    return lfw_images, lfw_labels

## reverse_images testing
# i1 = np.array([[1, 2, 3], [4, 5, 6]])
# i2 = np.array([[10, 20, 30], [40, 50, 60]])
#
# i1 = i1.flatten()
# i2 = i2.flatten()
#
# images = np.array([i1, i2])
# print(images.shape)
# print(reverse_images(images, (2, 3)))
