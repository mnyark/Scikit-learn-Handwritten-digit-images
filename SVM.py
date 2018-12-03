
# coding: utf-8

# In[5]:


get_ipython().magic('matplotlib inline')
import numpy as np
from sklearn.datasets import load_digits
digits = load_digits()
digits.images.shape
from scipy import stats
import matplotlib.pyplot as plt

#plot the dataset
fig, axes = plt.subplots(10, 10, figsize=(8, 8),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='binary', interpolation='nearest')
    ax.text(0.05, 0.05, str(digits.target[i]),
            transform=ax.transAxes, color='green')


X = digits.data
X.shape
y = digits.target
y.shape


#Kernel SVM
from mpl_toolkits import mplot3d
from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from sklearn.datasets.samples_generator import make_circles

X, y = make_circles(100, factor=.1, noise=.1)

def plot_3D(elev=30, azim=30, X=X, y=y):
    ax = plt.subplot(projection='3d')
    ax.scatter3D(X[:, 0], X[:, 1], r, c=y, s=50, cmap='autumn')
    ax.view_init(elev=None, azim=azim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('r')
r = np.exp(-(X ** 2).sum(1))
interact(plot_3D, elev=[-90, 90], azip=(-180, 180),
         X=fixed(X), y=fixed(y));

# SVC
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

pca = PCA(svd_solver='randomized', n_components=64, random_state=9)
svc = SVC(kernel='rbf', class_weight = 'balanced')
model = make_pipeline(pca, svc)

from sklearn.cross_validation import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(digits.data, digits.target, random_state=20)

from sklearn.grid_search import GridSearchCV
param_grid = {'svc__C': [1, 5, 10],
              'svc__gamma': [0.0001, 0.0005, 0.001]}

grid = GridSearchCV(model, param_grid)
get_ipython().magic('time grid.fit(Xtrain, ytrain)')
print(grid.best_params_)

model = grid.best_estimator_
yfit = model.predict(Xtest)



fig, ax = plt.subplots(10, 10, figsize=(8, 8),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))

for i, axi in enumerate(ax.flat):
    axi.imshow(Xtest[i].reshape(8,8), cmap='binary', interpolation='nearest')
    axi.text(0.05, 0.05, str(digits.target[yfit[i]]),
            transform=axi.transAxes, color='green' if yfit[i] == ytest[i] else 'red')


    
from sklearn import metrics
print(metrics.classification_report(ytest, yfit))

plt.show()

from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(ytest, yfit)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label');
acc = accuracy_score(ytest, yfit)
print("This is the accuracy score: ", acc)

