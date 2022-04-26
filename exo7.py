from sklearn import svm
from generateData import generate_random_dataset_xor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

normalisation = 0
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
p_grid = [{"C": [1, 10, 100, 500, 1000],
           "gamma": [.05, .1, .5, 1]}]
model = svm.SVC(kernel='rbf', class_weight='balanced')


grid = GridSearchCV(verbose=1, estimator=model, param_grid=p_grid, scoring=make_scorer(
    balanced_accuracy_score, needs_proba=False), cv=3)
grid.fit(x, y)
# fit estimates the best hyperparameters using cross-validation and then learns a model from the best hyperparameters using the whole training dataset.
# could be written as follows
# Pour la validation croisée, le principe est le suivant :
# on découpe l'ensemble d'apprentissage en 3 parties (cv=3). On les denotera P1,P2,P3
# Pour chaque hyperparametre possible, on fait les 2 etapes suivantes
# 1/On utilise P1 et P2 pour faire l'apprentissage et on utilise P3 pour evaluer
# le modele (avec la fonction balanced_accuracy_score). On obtient un score.
# 2/On utilise P1 et P3 pour faire l'apprentissage et on utilise P2 pour evaluer
# le modele (avec la fonction balanced_accuracy_score). On obtient un score.
# 3/On utilise P2 et P3 pour faire l'apprentissage et on utilise P1 pour evaluer
# le modele (avec la fonction balanced_accuracy_score). On obtient un score.
# on obtient au final un score moyen pour chaque jeu d'hyperparamètres
# on selectionne la valeur des hyperparametres qui permet d'obtenir le meilleur score moyen
# on utilise finalement P1,P2 et P3 pour faire l'apprentissage en utilisant les hyperparametres prealablement estimes
# les meilleurs paramètres sont grid.best_params_['C'], gamma=grid.best_params_['gamma']
y_pred = grid.predict(x2)
print(balanced_accuracy_score(y2, y_pred))
