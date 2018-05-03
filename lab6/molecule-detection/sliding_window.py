import classifier
import numpy as np
import preprocessing

from PIL import Image, ImageDraw

import os.path as path
import os

__DEFAULT_PATH = path.join(os.getcwd(), 'data')
__OUT_PATH     = path.join(os.getcwd(), 'out')
__IMAGE_PATH   = path.join(__DEFAULT_PATH, 'images')
__IMAGE_FILES  = os.listdir(__IMAGE_PATH)

__IMG_DIM = (64, 64)
__MOL_DIM = (7, 7)

def create_7x7_images(image64x64):
    images7x7 = []
    # coords    = []
    for i in range(image64x64.shape[0] - __MOL_DIM[0]):
        for j in range(image64x64.shape[1] - __MOL_DIM[1]):
            image7x7 = image64x64[i:i+__MOL_DIM[0], j:j+__MOL_DIM[1]]
            image7x7 = preprocessing.scale_image(image7x7)
            images7x7.append(image7x7)
            # coords.append((i, j))
    images7x7 = np.array(images7x7)
    images7x7 = images7x7.reshape((-1, images7x7.shape[1],
                                       images7x7.shape[2], 1))
    return images7x7


def detect_probabilities(model, images7x7, img_dim=__IMG_DIM, mol_dim=__MOL_DIM):
    probabilities = model.predict(images7x7)[:,1]
    return probabilities.reshape((__IMG_DIM[0] - __MOL_DIM[0],
                                  __IMG_DIM[1] - __MOL_DIM[1], 1))

def create_image(arr, path, coords=None):
    img = Image.fromarray(arr)

    if coords is not None:
        d = ImageDraw.Draw(img)
        d.rectangle(xy=[coords[0], coords[1], coords[0] + __MOL_DIM[0], coords[1] + __MOL_DIM[1]], outline=0)

    img.save(path)


def highest_probability_region(prob_map):
    AREA = 3
    max_prob  = 0.0
    max_coord = [0] * 2
    for i in range(prob_map.shape[0] - AREA):
        for j in range(prob_map.shape[0] - AREA):
            tmp = np.average(prob_map[i:i+AREA, j:j+AREA])
            if tmp > max_prob:
                max_prob = tmp
                max_coord[0] = i+(AREA//2)
                max_coord[1] = j+(AREA//2)
    return max_coord


images = preprocessing.load_images(scale=True)
model  = classifier.train_classifier()

for i, image64x64 in enumerate(images):
    images7x7 = create_7x7_images(image64x64)
    probabilities = detect_probabilities(model, images7x7)
    max_prob_coord = highest_probability_region(probabilities)
    probabilities = np.array(probabilities[:,:,0] * 255, dtype=np.uint8)

    prob_path = path.join(__OUT_PATH, str(i) + '_prob.png')
    bb_path = path.join(__OUT_PATH, str(i) + '_bb.png')

    image64x64 = image64x64[:,:,0]
    image64x64 = np.array(image64x64 * 255, dtype=np.uint8)

    create_image(arr=probabilities, path=prob_path)
    create_image(arr=image64x64, path=bb_path, coords=max_prob_coord)




# probabilities.shape
#
#
#
# original_image = np.array(image64x64*255, dtype=np.uint8)
#
# test = np.array(images7x7[1][:,:,0] *255, dtype=np.uint8)
#
# create_image(arr=probabilities, path='test1.png')
# create_image(arr=original_image, path='test2.png')
