import numpy as np
from sklearn import svm
from generateData import generate_random_dataset_linear_separable_noise
from fig import plot_points_with_label, plot_decision_regions
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler



def process(model,x,y):
    model.fit(x,y)    
    y_predict = model.predict(x)
    CM = confusion_matrix(y,y_predict,normalize='true')
    print("pourcentage de bonne classification de la premiere classe  : {}".format(CM[0][0]))
    print("pourcentage de bonne classification de la seconde classe   : {}".format(CM[1][1]))
    print("pourcentage moyen : {}".format(0.5*(CM[1][1]+CM[0][0])))
    
# Generate datasets
size = 200
dataset = generate_random_dataset_linear_separable_noise(size)
features = dataset[['x', 'y']]
label = dataset['target']
x = features.values
y = label.values
x[:,1]=x[:,1]*100;
x[:,0]=x[:,0]/100;
plot_points_with_label(x, y)


model = svm.SVC(kernel='linear',class_weight='balanced',C = 1, gamma=1)
print("without normalization")
process(model,x,y)

#normalisation des données
Scaler = StandardScaler().fit(X=x)
x = Scaler.transform(x)

print("with normalization")
print("moyenne de x : {}".format(-1)) # a changer
print("variance de x : {}".format(-1)) # a changer

model = svm.SVC(kernel='linear',class_weight='balanced',C = 1, gamma=1)
process(model,x,y)


# Generate datasets
size = 200
dataset = generate_random_dataset_linear_separable_noise(10)
features = dataset[['x', 'y']]
label = dataset['target']
x = features.values
y = label.values
x[:,1]=x[:,1]*100;
x[:,0]=x[:,0]/100;

print("testing new data")
x_1 = x #to change
x_2 = x #to change
print("difference maximale entre x_1 et x_2 : {}".format(np.max(np.abs(x_1-x_2))))
y_predict_1 = model.predict(x_1)
print(y_predict_1)
print(y)


