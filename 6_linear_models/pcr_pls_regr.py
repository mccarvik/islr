# perform standard imports
import sys
import pdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy
from matplotlib import pyplot as plt

from tqdm import tqdm # a python package that provides progress bars for iterables
from operator import itemgetter
from itertools import combinations
from sklearn.model_selection import KFold
# Standard libraries
from itertools import combinations

# Math and data processing
import scipy as sp

# StatsModels
import statsmodels.api as sm
import statsmodels.formula.api as smf

# scikit-learn
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold, cross_val_score
from sklearn.preprocessing import scale
from sklearn.metrics import mean_squared_error

# Visulization
from IPython.display import display
import matplotlib as mpl
import seaborn as sns
mpl.style.use('ggplot')
PATH = '/home/ec2-user/environment/islr/6_linear_models/figs/'

dummy_vars = ['League_N', 'Division_W', 'NewLeague_N']
response = 'Salary'
# Generate dummy variables for qualitative variables
qual_vars = ['League', 'Division', 'NewLeague']

hitters = pd.read_csv('Hitters.csv', index_col=0)
hitters.dropna(axis=0, inplace=True)
hitters_dummies = pd.get_dummies(hitters[qual_vars])
y = hitters[response]
# Drop response and qualitative variables, and combine with dummy data frame
X = pd.concat([hitters.drop(qual_vars + [response], axis=1), hitters_dummies[dummy_vars]], axis=1)
features = hitters.columns.drop([response])
print(X.info())

pca = PCA()
X_reduced = pca.fit_transform(scale(X))
print("The sahpe of loading matrix:", pca.components_.shape)
print("\nHead of loading matrix:")
print(pd.DataFrame(pca.components_.T).loc[:4,:5])

# PCR with number of PCs from 0 to 19, with 10-fold CV
# For 0 PC case, regression on intercept only.
kf = KFold(n_splits=10, shuffle=True, random_state=1)
regr = LinearRegression()
mse = []
for n_pc in range(0, pca.n_components_ + 1):
    if n_pc == 0:
        X_regr = np.ones((len(y),1))
    else:
        X_regr = X_reduced[:, :n_pc]
    scores = cross_val_score(regr, X_regr, y, cv=kf, scoring='neg_mean_squared_error')
    mse.append(scores.mean() * (-1))

# Find the n_pc with lowest MSE, or highest CV score
min_mse = min(mse)
min_mse_idx = mse.index(min_mse)    
    
# Plot MSE vs. number of PCs
fig, ax = plt.subplots(figsize=(8, 6))
plt.plot(mse)
plt.xticks(range(20), range(20))
min_mse_marker, = plt.plot(min_mse_idx, min_mse, 'b*', markersize=15)
plt.xlabel('Number of Principal Components')
plt.ylabel('MSE')
plt.title('Principal Component Regression with 10-Fold Cross-Validation')
plt.legend([min_mse_marker], ['Best number of principal components'])
plt.savefig(PATH + 'pca.png', dpi=300)
plt.close()

evr = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
print("Explained Variance Ratio:")
print(pd.Series([str(p) + ' %' for p in evr]))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
X_train_reduced = pca.fit_transform(scale(X_train))

# PCR on training set with number of PCs from 0 to 19, with 10-fold CV
# For 0 PC case, regression on intercept only.
kf = KFold(n_splits=10, shuffle=True, random_state=1)
regr = LinearRegression()
mse = []
for n_pc in range(0, pca.n_components_ + 1):
    if n_pc == 0:
        X_regr = np.ones((len(y_train),1))
    else:
        X_regr = X_train_reduced[:, :n_pc]
    scores = cross_val_score(regr, X_regr, y_train, cv=kf, scoring='neg_mean_squared_error')
    mse.append(scores.mean() * (-1))

# Find the n_pc with lowest MSE, or highest CV score
min_mse = min(mse)
min_mse_idx = mse.index(min_mse)    
    
# Plot MSE vs. number of PCs
fig, ax = plt.subplots(figsize=(8, 6))
plt.plot(mse)
plt.xticks(range(20), range(20))
min_mse_marker, = plt.plot(min_mse_idx, min_mse, 'b*', markersize=15)
plt.xlabel('Number of Principal Components')
plt.ylabel('MSE')
plt.title('Principal Component Regression on Training Set with 10-Fold Cross-Validation')
plt.legend([min_mse_marker], ['Best number of principal components'])
plt.savefig(PATH + 'pca_test.png', dpi=300)
plt.close()

X_test_reduced = pca.transform(scale(X_test))[:,:7]

# Train regression model on training data 
regr = LinearRegression()
regr.fit(X_train_reduced[:,:7], y_train)

# Prediction with test data
y_pred = regr.predict(X_test_reduced)
print("Test set MSE = ", mean_squared_error(y_test, y_pred))

# PARTIAL LEAST SQUARES
kf = KFold(n_splits=10, shuffle=True, random_state=1)
mse = []
for i in range(1, 20):
    pls = PLSRegression(n_components=i)
    scores = cross_val_score(pls, scale(X_train), y_train, cv=kf, scoring='neg_mean_squared_error')
    mse.append(scores.mean() * (-1))

# Find the number of PLS directions with lowest MSE, or highest CV score
min_mse = min(mse)
min_mse_idx = mse.index(min_mse) + 1

# Plot MSE vs. number of PLS directions
fig, ax = plt.subplots(figsize=(8, 6))
plt.plot(range(1, 20), mse)
plt.xticks(range(20), range(20))
min_mse_marker, = plt.plot(min_mse_idx, min_mse, 'b*', markersize=15)
plt.xlabel('Number of PLS directions')
plt.ylabel('MSE')
plt.title('PLS on Training Set with 10-Fold Cross-Validation')
plt.legend([min_mse_marker], ['Best number of PLS directions'])
plt.savefig(PATH + 'pca_pls.png', dpi=300)
plt.close()


pls = PLSRegression(n_components=2)
pls.fit(scale(X_train), y_train)
print(mean_squared_error(y_test, pls.predict(scale(X_test))))
