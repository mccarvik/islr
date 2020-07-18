import itertools

import pdb
import numpy as np
import pandas as pd
import patsy as pt
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import GridSearchCV
PATH = '/home/ec2-user/environment/islr/9_support_vector_machines/figs/'

# Generate set that is linearly seperable
np.random.seed(1)
X = np.random.normal(0, 1, (20, 2))
y = np.concatenate((np.repeat(0, 10), np.repeat(1, 10)))
X[y==1, :] = X[y==1, :] + 1.3

df = pd.concat([pd.DataFrame(data=X, columns=['x1', 'x2']), pd.Series(y, name='y')], axis=1)
plt.figure(figsize=(10, 10))
sns.scatterplot(x='x1', y='x2', hue='y', data=df)
plt.savefig(PATH + 'lin_sep_rand.png', dpi=300)
plt.close()

model = svm.SVC(kernel='linear', C=1e5, random_state=0).fit(X, y)
# Decistion boundary plot
x1 = np.arange(-.7, 3.5, .1)
x2 = np.arange(-2.5, 3, .1)
xx1, xx2 = np.meshgrid(x1, x2, sparse=False)

Xgrid = np.stack((xx1.flatten(), xx2.flatten())).T
y_hat = model.predict(Xgrid)
y_grid = y_hat.reshape(len(x2), len(x1))
y_grid.shape

fig = plt.figure(figsize=(10, 10))
plt.contourf(x1, x2, y_grid);
sns.scatterplot(x='x1', y='x2', hue='y', data=df)
sns.scatterplot(x=model.support_vectors_[:,0], y=model.support_vectors_[:,1], color='red', marker='+', s=500);
plt.savefig(PATH + 'lin_sep_model.png', dpi=300)
plt.close()

print('Model parameters:')
print(model.get_params)

print('\nNumber of support vectors for each class.:')
print(model.n_support_)

print('\nCoefficients of the support vector in the decision function. :')
print(model.dual_coef_)

print('\nWeights assigned to the features (coefficients in the primal problem).')
print(model.coef_)

# Smaller cost - linear seperable
model = svm.SVC(kernel='linear', C=1, random_state=0).fit(X, y)
# Decistion boundary plot
x1 = np.arange(-.7, 3.5, .1)
x2 = np.arange(-2.5, 3, .1)
xx1, xx2 = np.meshgrid(x1, x2, sparse=False)

Xgrid = np.stack((xx1.flatten(), xx2.flatten())).T
y_hat = model.predict(Xgrid)
y_grid = y_hat.reshape(len(x2), len(x1))
y_grid.shape

fig = plt.figure(figsize=(10, 10))
plt.contourf(x1, x2, y_grid);
sns.scatterplot(x='x1', y='x2', hue='y', data=df)
sns.scatterplot(x=model.support_vectors_[:,0], y=model.support_vectors_[:,1], color='red', marker='+', s=500);
plt.savefig(PATH + 'lin_sep_small_cost_model.png', dpi=300)
plt.close()

print('Model parameters:')
print(model.get_params)

print('\nNumber of support vectors for each class.:')
print(model.n_support_)

print('\nCoefficients of the support vector in the decision function. :')
print(model.dual_coef_)

print('\nWeights assigned to the features (coefficients in the primal problem).')
print(model.coef_)
