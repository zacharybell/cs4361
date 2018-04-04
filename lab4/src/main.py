from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, Normalizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion

from transformer import ModelTransformer

import numpy as np


RANDOM = 99

X = dict()
y = dict()

X_train = dict()
X_test = dict()
y_train = dict()
y_test = dict()


# Package preprocessors and estimators so they are easier to use

preprocessors = {
    'std': StandardScaler(),
    'norm': Normalizer(),
    'pca': PCA()
}

estimators = {
    'knn': [
        ModelTransformer(KNeighborsClassifier()),
        { 'n_neighbors': [1, 3, 5, 9], 'p': [1, 2] }
    ],
    'mlp': [
        ModelTransformer(MLPClassifier(random_state=RANDOM)),
        {
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'alpha': [0.000001, 0.00001, 0.0001, 0.001, 0.01]
        }
    ],
    'svm': [
        ModelTransformer(SVC(random_state=RANDOM)),
        { 'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'] }
    ],
    'tree': [
        ModelTransformer(DecisionTreeClassifier(random_state=RANDOM)),
        {
            'criterion': ['gini', 'entropy'],
            'min_sample_leaf': [1, 3, 5, 9],
            'min_impurity_decrease': [0, 0.1, 0.25, 0.5]
        }
    ],
    'forest': [
        ModelTransformer(RandomForestClassifier(random_state=RANDOM)),
        {
            'criterion': ['gini', 'entropy'],
            'min_sample_leaf': [1, 3, 5, 9],
            'min_impurity_decrease': [0, 0.1, 0.25, 0.5]
        }
    ]
}


# Read in molecules data and create class labels (0 and 1)

mol = np.load('./lab4/res/molecules.npy').reshape((-1,49))
no_mol = np.load('./lab4/res/no_mol.npy').reshape((-1,49))

n_ones = mol.shape[0]
n_zeros = no_mol.shape[0]

X['mol'] = np.concatenate((mol, no_mol), axis=0)
y['mol'] = np.append(np.ones(n_ones, dtype=int), np.zeros(n_zeros, dtype=int))


# Read in RNA data

X['rna'] = np.load('./lab4/res/rnafolding_X.npy')
y['rna'] = np.load('./lab4/res/rnafolding_y.npy').ravel()


# Create shuffled train and test sets

X_train['mol'], X_test['mol'], y_train['mol'], y_test['mol'] = train_test_split(
    X['mol'], y['mol'], test_size=0.2, shuffle=True, random_state=RANDOM
)

X_train['rna'], X_test['rna'], y_train['rna'], y_test['rna'] = train_test_split(
    X['rna'], y['rna'], test_size=0.2, shuffle=True, random_state=RANDOM
)



def conglomeration(preprocessors, estimators):
    pipes = list()

    for p_key in preprocessors:
        for e_key in estimators:
            preprocessor = (p_key, preprocessors[p_key])
            estimator = (e_key, estimators[e_key])
            pipe = create_searchable_pipeline(preprocessor, estimator)

            pipes.append((p_key + '_' + e_key, pipe))

    return FeatureUnion(pipes)



# def create_searchable_pipeline(preprocessor, estimator):
#     parameters = estimator[1][1]
#     pipeline_parameters = dict()
#
#     for key in parameters:
#         pipeline_parameters[estimator[0] + '__' + key] = parameters[key]
#
#     pipe = Pipeline([(preprocessor[0], preprocessor[1]), (estimator[0], estimator[1][0])])
#
#     return GridSearchCV(pipe, param_grid=pipeline_parameters, cv=5)


def create_searchable_pipeline(preprocessor, estimator):
    parameters = estimator[1][1]
    pipeline_parameters = dict()

    for key in parameters:
        pipeline_parameters[estimator[0] + '__' + key] = parameters[key]

    grid = GridSearchCV(estimator=estimator[1][0], param_grid=estimator[1][1], cv=5)

    return Pipeline([(preprocessor[0], preprocessor[1]), (estimator[0], grid)])


congl = conglomeration(preprocessors, estimators)



# pipeline = create_searchable_pipeline(
# ('std', StandardScaler()),
# ('knn', [KNeighborsClassifier(), { 'n_neighbors': [1, 3, 5, 9], 'p': [1, 2] }])
# )
#
# print(pipeline)
