import numpy as np
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm
from generateData import unbalanced_data
from fig import plot_points_with_label, plot_decision_regions
from sklearn.metrics import confusion_matrix

def process(model,x,y):
    model.fit(x,y)    
    plot_decision_regions(x, y, classifier=model)    
    y_predict = model.predict(x)
    CM = confusion_matrix(y,y_predict,normalize='true')
    print("Matrice de confusion normalisee")
    print(CM)
    print("pourcentage de bonne classification de la premiere classe  : {}".format(CM[0][0]))
    print("pourcentage de bonne classification de la seconde classe   : {}".format(CM[1][1]))
    print("pourcentage moyen : {}".format(0.5*(CM[1][1]+CM[0][0])))
    print("Matrice de confusion non normalisee")
    CM = confusion_matrix(y,y_predict,normalize=None)
    print(CM)
    print("pourcentage de bonne classification : {}".format((CM[1][1]+CM[0][0])/(np.sum(CM))))
    
# Generate dataset
size = 200
dataset = unbalanced_data(100,10)
features = dataset[['x', 'y']]
label = dataset['target']
x = features.values
y = label.values
plot_points_with_label(x, y)

# SVM RBF
model = svm.SVC(kernel='linear')
process(model,x,y)
    
   
    

