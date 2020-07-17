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

# Generate som sample data with 2 predictors and a response
np.random.seed(2)
X = np.random.normal(0, 1, (20, 2))
y = np.concatenate((np.repeat(0, 10), np.repeat(1, 10)))
X[y==1, :] = X[y==1, :] +1

# Plot it, is it linearly seperable?
df = pd.concat([pd.DataFrame(data=X, columns=['x1', 'x2']), pd.Series(y, name='y')], axis=1)
print(df)
plt.figure(figsize=(10, 10))
sns.scatterplot(x='x1', y='x2', hue='y', data=df);
plt.savefig(PATH + 'rand.png', dpi=300)
plt.close()

model = svm.SVC(kernel='linear', C=10, random_state=0).fit(X, y)
# Decistion boundary plot
x1 = np.arange(-4, 4, .1)
x2 = np.arange(-4, 4, .1)
xx1, xx2 = np.meshgrid(x1, x2, sparse=False)

Xgrid = np.stack((xx1.flatten(), xx2.flatten())).T
y_hat = model.predict(Xgrid)
y_grid = y_hat.reshape(len(x2), len(x1))
y_grid.shape

fig = plt.figure(figsize=(10, 10))
plt.contourf(x1, x2, y_grid);
sns.scatterplot(x='x1', y='x2', hue='y', data=df)
sns.scatterplot(x=model.support_vectors_[:,0], y=model.support_vectors_[:,1], color='red', marker='+', s=500);
plt.savefig(PATH + 'pred_on_rand.png', dpi=300)
plt.close()

# Contour plot
x1 = np.arange(-4, 4, .1)
x2 = np.arange(-4, 4, .1)
xx1, xx2 = np.meshgrid(x1, x2, sparse=False)

Xgrid = np.stack((xx1.flatten(), xx2.flatten())).T
y_hat = model.decision_function(Xgrid)
y_grid = y_hat.reshape(len(x2), len(x1))
y_grid.shape

fig = plt.figure(figsize=(10, 10))
plt.contourf(x1, x2, y_grid);
sns.scatterplot(x='x1', y='x2', hue='y', data=df);
sns.scatterplot(x=model.support_vectors_[:,0], y=model.support_vectors_[:,1], color='black', marker='+', s=500)
plt.savefig(PATH + 'contour_pred_on_rand.png', dpi=300)
plt.close()

# We can obtain some basic information about the support vector classifier
print('Model parameters:')
print(model.get_params)

print('\nNumber of support vectors for each class.:')
print(model.n_support_)

print('\nCoefficients of the support vector in the decision function. :')
print(model.dual_coef_)

print('\nWeights assigned to the features (coefficients in the primal problem).')
print(model.coef_)

# SMALLER COST PARAMETER
model = svm.SVC(kernel='linear', C=0.1, random_state=0).fit(X, y)
# Decistion boundary plot
x1 = np.arange(-4, 4, .1)
x2 = np.arange(-4, 4, .1)
xx1, xx2 = np.meshgrid(x1, x2, sparse=False)

Xgrid = np.stack((xx1.flatten(), xx2.flatten())).T
y_hat = model.predict(Xgrid)
y_grid = y_hat.reshape(len(x2), len(x1))
y_grid.shape

fig = plt.figure(figsize=(10, 10))
plt.contourf(x1, x2, y_grid);
sns.scatterplot(x='x1', y='x2', hue='y', data=df)
sns.scatterplot(x=model.support_vectors_[:,0], y=model.support_vectors_[:,1], color='red', marker='+', s=500);
plt.savefig(PATH + 'smaller_cost_pred.png', dpi=300)
plt.close()

print('Model parameters:')
print(model.get_params)

print('\nNumber of support vectors for each class.:')
print(model.n_support_)

print('\nCoefficients of the support vector in the decision function. :')
print(model.dual_coef_)

print('\nWeights assigned to the features (coefficients in the primal problem).')
print(model.coef_)

# Use cross-validation to tune Cost parameter
Cs = [0.001, 0.01, 0.1, 1, 5, 10, 100]
scores = []
for C in Cs:
    model = svm.SVC(kernel='linear', C=C, random_state=0)
    score = cross_val_score(model, X, y, cv=5)
    scores += [score]
scores_mean = np.mean(np.asarray(scores), axis=1)
pd.DataFrame({'C': Cs, 'accuracy': scores_mean})
plt.figure(figsize=(10,10))
sns.lineplot(x=np.log(Cs), y=scores_mean)
plt.xlabel('log(C)')
plt.savefig(PATH + 'cv_cost_param.png', dpi=300)
# plt.ylabel('accuracy');
plt.close()

# Generate simulated test set
xtest = np.random.normal(0, 1, (20, 2))
ytest = np.random.choice([0, 1], size=20, replace=True)
xtest[ytest==1, :] = xtest[ytest==1, :] +1
# With cost = 5
# Test model selected by cross-validation
model = svm.SVC(kernel='linear', C=5, random_state=0).fit(X, y)
ypred = model.predict(xtest)

print(confusion_matrix(ytest, ypred))
# With cost = 0.01
# Test model selected by cross-validation
model = svm.SVC(kernel='linear', C=0.01, random_state=0).fit(X, y)
ypred = model.predict(xtest)

print(confusion_matrix(ytest, ypred))
