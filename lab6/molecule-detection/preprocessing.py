import numpy as np
import os.path as path
import os

from sklearn.preprocessing import MinMaxScaler

__DEFAULT_PATH = path.join(os.getcwd(), 'data')
__MOL_PATH     = path.join(__DEFAULT_PATH, 'molecules.npy')
__NO_MOL_PATH  = path.join(__DEFAULT_PATH, 'no_mol.npy')

def scale_image(image):
    min = np.min(image)
    max = np.max(image)
    image = (image - min) / (max - min)
    return image

def load_mol_data(mol_path=__MOL_PATH, no_mol_path=__NO_MOL_PATH):
    mol = np.load(mol_path)
    no_mol = np.load(no_mol_path)
    images = np.append(mol, no_mol, axis=0)
    images = np.apply_along_axis(scale_image, axis=1, arr=images)
    images = images.reshape((-1, images.shape[1], images.shape[2],1))
    labels = np.ones(mol.shape[0], dtype=np.int8)
    labels = np.append(labels, np.zeros(no_mol.shape[0], dtype=np.int8))
    return images, labels
