import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

def versiontuple(v):
    return tuple(map(int, (v.split("."))))

def plot_points_with_label(X, y):
    #X.shape[0] ou y.shape[0] #nbre d echantillons
    #X.shape[1] #nbre de caracteristiques
    #len(np.unique(y)) : nbre de classes

    nbreClasse = len(np.unique(y))
    if nbreClasse > 5:
        return;
    
    if X.shape[1] > 3:
        return;
                
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    if X.shape[1] == 2:
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],alpha=0.8, c=colors[idx],marker=markers[idx], label=cl)   
    else :
        fig = plt.figure()
        ax = Axes3D(fig)
        for idx, cl in enumerate(np.unique(y)):
            ax.scatter(xs=X[y == cl, 0], ys=X[y == cl, 1],zs=X[y == cl, 2],alpha=0.8, c=cmap(idx),marker=markers[idx], label=cl)
    
    plt.show()
    
def plot_decision_regions(X, y, classifier, resolution=0.1):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))]) 

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):        
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],alpha=0.8, c=ListedColormap(('red', 'blue', 'lightgreen', 'gray', 'cyan'))(idx),marker=markers[idx], label=cl)
        

         
    if isinstance(classifier,svm.SVC):
        xy = np.vstack([xx1.ravel(), xx2.ravel()]).T
        Z = classifier.decision_function(xy).reshape(xx1.shape)
        
        #Draw the decision boundary and margins
        plt.contour(xx1, xx2, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
        if classifier.support_vectors_.shape[0]>0:
            plt.scatter(classifier.support_vectors_[:, 0], classifier.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')
    plt.show()
    







