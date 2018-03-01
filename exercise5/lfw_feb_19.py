from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
import numpy as np

people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape

# fig, axes = plt.subplots(2, 5, figsize=(15, 8),
#                          subplot_kw={'xticks': (), 'yticks': ()})
# for target, image, ax in zip(people.target, people.images, axes.ravel()):
#     ax.imshow(image, cmap='gray')
#     ax.set_title(people.target_names[target])
#
# print("people.images.shape: {}".format(people.images.shape))
# print("Number of classes: {}".format(len(people.target_names)))


# # count how often each target appears
# counts = np.bincount(people.target)
# # print counts next to target names:
# for i, (count, name) in enumerate(zip(counts, people.target_names)):
#     print("{0:25} {1:3}".format(name, count), end='   ')
#     if (i + 1) % 3 == 0:
#         print()
# print()

mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1

X_people = people.data[mask]
y_people = people.target[mask]

# scale the grey-scale values to be between 0 and 1
# instead of 0 and 255 for better numeric stability:
X_people = X_people / 255.


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
# split the data in training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X_people, y_people, stratify=y_people, random_state=0)
# build a KNeighborsClassifier with using one neighbor:
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print("Test set score of 1-nn: {:.3f}".format(knn.score(X_test, y_test)))
