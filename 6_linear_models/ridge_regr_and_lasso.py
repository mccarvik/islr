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

# Lab 2: Ridge Regression and the Lasso

hitters = pd.read_csv('Hitters.csv', index_col=0)
hitters.dropna(axis=0, inplace=True)
# Generate dummy variables for qualitative variables
qual_vars = ['League', 'Division', 'NewLeague']
hitters_dummies = pd.get_dummies(hitters[qual_vars])
print(hitters_dummies.info())

# Define X, y features and reponse data for scikit-learn
dummy_vars = ['League_N', 'Division_W', 'NewLeague_N']
response = 'Salary'
y = hitters[response]
# Drop response and qualitative variables, and combine with dummy data frame
X = pd.concat([hitters.drop(qual_vars + [response], axis=1), hitters_dummies[dummy_vars]], axis=1)
features = hitters.columns.drop([response])
print(X.info())

# Ridge regression on full dataset, over 100 alphas from 10 to -2
alphas = 10**np.linspace(10,-2,100)
ridge = Ridge()
coefs = []
for alpha in alphas:
    ridge.set_params(alpha=alpha*0.5)  # alpha/2 to align with R
    ridge.fit(scale(X), y)
    coefs.append(ridge.coef_)
# Plot
fig, ax = plt.subplots(figsize=(8,6))
plt.plot(alphas, coefs)
ax.set_xscale('log')
plt.xlim(1e10, 1e-2)  # reverse axis
plt.xlabel('Regularization Parameter')
plt.ylabel('Coefficients')
plt.title('Ridge regression coefficients vs. regularization parameter.')
plt.savefig(PATH + 'ride_regr.png', dpi=300)
plt.close()

# Split Hitters data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
# Fit on training set with alpha = 4
ridge_a4 = Ridge(alpha=4, normalize=True)
ridge_a4.fit(X_train, y_train)
y_pred_a4 = ridge_a4.predict(X_test)
print('\nRidge regression coefficients:\n', pd.Series(ridge_a4.coef_, index=features))
print('\nMSE = ', mean_squared_error(y_test, y_pred_a4))

# Fit on training set with alpha = 10^10
ridge_a1e10 = Ridge(alpha=10**10, normalize=True)
ridge_a1e10.fit(X_train, y_train)
y_pred_a1e10 = ridge_a1e10.predict(X_test)
print('\nRidge regression coefficients:\n', pd.Series(ridge_a1e10.coef_, index=features))
print('\nMSE = ', mean_squared_error(y_test, y_pred_a1e10))

# Ridge regularization with cross-validation
ridgecv = RidgeCV(alphas=alphas*0.5, scoring='neg_mean_squared_error', normalize=True)
ridgecv.fit(X_train, y_train)
print("The best ridge regularization Alpha = ", ridgecv.alpha_)
ridge_cv = Ridge(alpha=ridgecv.alpha_, normalize=True)
ridge_cv.fit(X_train, y_train)
mse = mean_squared_error(y_test, ridge_cv.predict(X_test))
print("MSE = ", mse)

# Ridge regression on full data set with CV-optimized alpha
ridge_cv.fit(X, y)
print(pd.Series(ridge_cv.coef_, index=features))

# LASSO
# Lasso regression on training set, over 100 alphas from 10 to -2.
lasso = Lasso(max_iter=10000, normalize=True)
coefs = []
for alpha in alphas:
    lasso.set_params(alpha=alpha)
    lasso.fit(scale(X_train), y_train)
    coefs.append(lasso.coef_)

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
plt.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])
plt.xlabel('Regularization Parameter')
plt.ylabel('Regression Coefficients')
plt.title('Lasso regression coefficients vs. regularization parameter.')
plt.savefig(PATH + 'lasso.png', dpi=300)
plt.close()

# Lasso regularization with cross-validation
lassocv = LassoCV(alphas=None, cv=10, max_iter=100000, normalize=True)
lassocv.fit(X_train, y_train)
print("The best Lasso regularization Alpha = ", ridgecv.alpha_)
pdb.set_trace()
lasso_cv = Lasso(max_iter=10000, normalize=True, alpha=lassocv.alpha_)
lasso_cv.fit(X_train, y_train)
mse = mean_squared_error(y_test, lasso_cv.predict(X_test))
print("MSE = ", mse)
print(pd.Series(lasso_cv.coef_, index=features))
