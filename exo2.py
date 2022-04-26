from sklearn import svm
from generateData import generate_random_dataset_linear_separable,generate_random_dataset_linear_separable_noise
from fig import plot_points_with_label, plot_decision_regions
from sklearn.metrics import confusion_matrix

# Generate dataset
size = 100
dataset = generate_random_dataset_linear_separable_noise(size)
features = dataset[['x', 'y']]
label = dataset['target']

x = features.values
y = label.values
plot_points_with_label(x, y)

CList = [0.0001,1,100,10000]
for c in CList:
    model = svm.SVC(kernel='linear',class_weight='balanced',C=c) # pour la question 4
    model.fit(x,y)    
    plot_decision_regions(x, y, classifier=model)    
    y_predict = model.predict(x)
    CM = confusion_matrix(y,y_predict,normalize=None)
    print("-------- valeur de C : {} ----------".format(c))
    print("Matrice de confusion (pas de normalisation) ")
    print(CM)
    
    CM = confusion_matrix(y,y_predict,normalize='true')
    print("Matrice de confusion (normalisation)  ")
    print(CM)
    print("pourcentage de bonne classification de la premiere classe  : {}".format(CM[0][0]))
    print("pourcentage de bonne classification de la seconde classe   : {}".format(CM[1][1]))
    print("pourcentage moyen : {}".format(0.5*(CM[1][1]+CM[0][0])))

 


