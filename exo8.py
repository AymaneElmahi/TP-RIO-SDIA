from math import factorial
import numpy as np
from sklearn import svm
from generateData import generate_random_dataset_xor
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

normalisation = 1
# Generate training dataset
size = 200
dataset = generate_random_dataset_xor(size)
features = dataset[['x', 'y']]
label = dataset['target']
x = features.values
y = label.values


# generate testing data
size = 200
dataset = generate_random_dataset_xor(size)
features = dataset[['x', 'y']]
label = dataset['target']
x2 = features.values
y2 = label.values

if normalisation == 1:
    x2[:, 1] = x2[:, 1]*100
    x2[:, 0] = x2[:, 0]/100
    x[:, 1] = x[:, 1]*100
    x[:, 0] = x[:, 0]/100

# Set up possible values of parameters to optimize over
p_grid = {"model__C": [1, 10, 100, 500, 1000],
          "model__gamma": [.05, .1, .5, 1]}
model = svm.SVC(kernel='rbf', class_weight='balanced')
new_model = Pipeline([('scaler', StandardScaler()), ('model', model)])

res = []
for i in range(5):
    skf = StratifiedKFold(n_splits=3, shuffle=True)
    for train_index, test_index in skf.split(x, y):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        grid = GridSearchCV(estimator=new_model, param_grid=p_grid, scoring=make_scorer(
            balanced_accuracy_score, needs_proba=False), cv=3)
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)
        res.append(balanced_accuracy_score(y_test, y_pred))

print("expected balanced accuracy {} with std {}".format(np.mean(res), np.std(res)))


# if we need a specific model
grid = GridSearchCV(estimator=new_model, param_grid=p_grid, scoring=make_scorer(
    balanced_accuracy_score, needs_proba=False), cv=3)
grid.fit(x, y)


# in the future, we have new data !!!
size = 200
y_pred = grid.predict(x2)
print("and after having new data, we obtain a score of {}".format(
    balanced_accuracy_score(y2, y_pred)))
