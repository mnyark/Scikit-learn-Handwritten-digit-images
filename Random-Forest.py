
# coding: utf-8

# In[1]:


get_ipython().magic('matplotlib inline')
import numpy as np
from sklearn.datasets import load_digits
digits = load_digits()
digits.keys()
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
#plot the dataset
fig, axes = plt.subplots(10, 10, figsize=(8, 8),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='binary', interpolation='nearest')
    ax.text(0.05, 0.05, str(digits.target[i]),
            transform=ax.transAxes, color='green')
plt.show()

X = digits.data
X.shape
y = digits.target
y.shape

# Random forest Classifier
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier

Xtrain, Xtest, ytrain, ytest = train_test_split(digits.data, digits.target,
                                                random_state=0)
model = RandomForestClassifier(n_estimators=1000)
model.fit(Xtrain, ytrain)
ypred = model.predict(Xtest)

from sklearn import metrics
print(metrics.classification_report(ypred, ytest))

from sklearn.metrics import confusion_matrix
mat = confusion_matrix(ytest, ypred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label');

fig, ax = plt.subplots(10, 10, figsize=(8, 8),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))

for i, axi in enumerate(ax.flat):
    axi.imshow(Xtest[i].reshape(8,8), cmap='binary', interpolation='nearest')
    axi.text(0.05, 0.05, str(digits.target[ypred[i]]),
            transform=axi.transAxes, color='green' if ypred[i] == ytest[i] else 'red')

