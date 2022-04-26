import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from generateData import generate_random_dataset_xor_high_dimension
from sklearn.metrics import make_scorer, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, RFE, RFECV
import time

# Generate dataset
size = 3000
dataset = generate_random_dataset_xor_high_dimension(size)

columns = []
for i in range(20):
    columns.append('x'+str(i))

features = dataset[columns]
label = dataset['target']
x = features.values
y = label.values

forest = RandomForestClassifier(
    n_estimators=40, class_weight='balanced_subsample', criterion='entropy', oob_score=True)
forest.fit(x, y)

importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(x.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


t1 = time.time()
s = SelectFromModel(forest, prefit=True, max_features=3)
x_new = s.transform(x)
print(x_new.shape)
print(x.shape)
print("time for SelectFromModel: {}".format(time.time()-t1))

t1 = time.time()
forest = RandomForestClassifier(
    n_estimators=40, class_weight='balanced_subsample', criterion='entropy', oob_score=True)
s = RFE(forest, n_features_to_select=3)
s = s.fit(x, y)
print(s.support_)
print(s.ranking_)
print("time for RFE: {}".format(time.time()-t1))

t1 = time.time()
s = RFECV(forest, cv=3, scoring=make_scorer(
    balanced_accuracy_score, needs_proba=False))
s = s.fit(x, y)
print(s.support_)
print(s.ranking_)
print("time for RFECV: {}".format(time.time()-t1))
print(s.cv_results_)  # or s.grid_scores_
# SequentialFeatureSelector from version 0.24
