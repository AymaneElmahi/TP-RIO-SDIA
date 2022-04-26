import numpy as np
from generateData import generate_random_dataset_high_dimension
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# Generate dataset
size = 100
dataset = generate_random_dataset_high_dimension(size)

columns = []
for i in range(10000):
    columns.append('x'+str(i))

features = dataset[columns]
label = dataset['target']
x = features.values
y = label.values

forest = RandomForestClassifier(n_estimators=1000, class_weight='balanced_subsample',
                                criterion='entropy', min_samples_split=20, max_features='sqrt', max_samples=0.4,)
forest.fit(x, y)
model = SelectFromModel(forest, prefit=True)
x_new = model.transform(x)
print(x_new.shape)
print(x.shape)

res = []
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=3)
for train_index, test_index in skf.split(x, y):
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    forest = RandomForestClassifier(n_estimators=100, class_weight='balanced_subsample', criterion='entropy',
                                    min_samples_split=20, max_features='sqrt', max_samples=0.4,)

    forest.fit(model.transform(X_train), y_train)
    y_pred = forest.predict(model.transform(X_test))
    res.append(balanced_accuracy_score(y_test, y_pred))

print(np.mean(res))
print(np.std(res))
