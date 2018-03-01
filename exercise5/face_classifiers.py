from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
import numpy as np

people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)

# mask applied to remove people with more than 50 pictures
mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1

X_people = people.data[mask]
y_people = people.target[mask]

# scale the grey-scale values to be between 0 and 1
# instead of 0 and 255 for better numeric stability:
X_people = X_people / 255.

# KNN (1 neighbor)
X_train, X_test, y_train, y_test = train_test_split(X_people,
                                                    y_people,
                                                    random_state=99)

# take in the hyperparams an perform best cross validation based on the data
def classification_suite(X_train, y_train, hyperparams):
    knn = KNeighborsClassifier()
    svm = SVC()
    tree = DecisionTreeClassifier()
    mlp = MLPClassifier(early_stopping=True)

    models = {}
    models['knn'] = GridSearchCV(knn, hyperparams['knn'])
    models['svm'] = GridSearchCV(svm, hyperparams['svm'])
    models['tree'] = GridSearchCV(tree, hyperparams['tree'])
    models['mlp'] = GridSearchCV(mlp, hyperparams['mlp'])

    for key in models:
        models[key].fit(X_train, y_train)

    return models

def classification_suite_score(models, X_test, y_test):
    score = {}
    for key in models:
        score[key] = models[key].score(X_test, y_test)
    return score

hyperparams = {
    'knn':{'n_neighbors':[1, 3, 9], 'weights':['uniform', 'distance']},
    'svm':{'kernel': ['linear', 'poly', 'rbf', 'sigmoid']},
    'tree':{'max_depth': [5, 8, 11, 14]},
    'mlp':{
        'alpha': [0.00001, 0.0001, 0.001],
        'hidden_layer_sizes': [(100,), (200,), (100, 100)]
        }
}

classification_suite(X_train, y_train, hyperparams)
