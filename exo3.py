import numpy as np
from sklearn import svm
from generateData import generate_random_dataset_circle
from fig import plot_points_with_label, plot_decision_regions
from sklearn.metrics import confusion_matrix

def process(model,x,y):
    print("model fitting")
    model.fit(x,y)    
    print("model prediction")
    y_predict = model.predict(x)
    CM = confusion_matrix(y,y_predict,normalize='true')
    print("pourcentage de bonne classification de la premiere classe  : {}".format(CM[0][0]))
    print("pourcentage de bonne classification de la seconde classe   : {}".format(CM[1][1]))
    print("pourcentage moyen : {}".format(0.5*(CM[1][1]+CM[0][0])))
    print("plotting decision region")
    plot_decision_regions(x, y, classifier=model)    
    
# Generate dataset
size = 200
dataset = generate_random_dataset_circle(size)
features = dataset[['x', 'y']]
label = dataset['target']
x = features.values
y = label.values
plot_points_with_label(x, y)


# SVM Lineaire
model = svm.SVC(kernel='linear',class_weight='balanced')
process(model,x,y)

# SVM RBF
model = svm.SVC(kernel='rbf',class_weight='balanced')
process(model,x,y)

# SVM avec noyau adapte
plot_points_with_label(x, y)
pts = np.empty((x.shape[0],3))
for i in range(x.shape[0]):
    pts[i][0] = x[i][0]*x[i][0]
    pts[i][1] = x[i][1]*x[i][1]
    pts[i][2] = np.sqrt(2) * x[i][0] * x[i][1]
plot_points_with_label(pts, y)  
#On peut observer qu'appliquer la transformation permet d'obtenir deux classes linéairement séparables. 
#On va coder cela en définissant un noyau adapté aux données (cf cours)

def my_kernel(X, Y):
    """
    We create a custom kernel:

                 (1  0)
    k(X, Y) = X  (    ) Y.T
                 (0  1)
    #M = np.array([[1, 0], [0, 1.0]])
    #return np.dot(np.dot(X, M), Y.T)    
    """
    a = np.dot(X, (Y.T))
    print("taille de X {}".format(X.shape))
    print("taille de Y {}".format(Y.shape))
    print("taille de a {}".format(a.shape)) 
    return np.multiply(a,a)
    
model = svm.SVC(kernel=my_kernel,class_weight='balanced')
process(model,x,y)

print("Predicting new data (usefull to see the call of my_kernel)")
dataset = generate_random_dataset_circle(10)
features = dataset[['x', 'y']]
label = dataset['target']
x = features.values
y = label.values
y_predict = model.predict(x)
print(y_predict)
print(y)


