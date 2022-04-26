import numpy as np
import matplotlib.pyplot as plt
from generateData import generate_random_dataset_xor_high_dimension
from sklearn.metrics import make_scorer,balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier


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

forest=RandomForestClassifier(n_estimators=400,class_weight='balanced_subsample',criterion='entropy',oob_score=True)

forest.fit(x, y) 
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(x.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the impurity-based feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(x.shape[1]), importances[indices],color="r", yerr=std[indices], align="center")
plt.xticks(range(x.shape[1]), indices)
plt.xlim([-1, x.shape[1]])
plt.show()

print("expected generalization error without using validation or testing data {}".format(forest.oob_score_))


size = 200
dataset = generate_random_dataset_xor_high_dimension(size)
features = dataset[columns]
label = dataset['target']
x2 = features.values
y2 = label.values
y_pred = forest.predict(x2);
print("and with using testing data : {}".format(balanced_accuracy_score(y2,y_pred)))



