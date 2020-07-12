import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn.linear_model as skl_lm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold, cross_val_score
from sklearn.preprocessing import PolynomialFeatures

import statsmodels.formula.api as smf
import statsmodels.api as sm

auto = pd.read_csv('auto.csv')
auto = auto[['mpg', 'horsepower']]
auto = auto.replace({'?': np.nan}).dropna().astype(float)
print(auto.head())

X = auto.horsepower
# Generating Polynomial Features
poly = PolynomialFeatures(2)
X_poly = poly.fit_transform(X.values.reshape(-1, 1))
X_train, X_test, y_train, y_test = train_test_split(X_poly, auto.mpg.ravel(),test_size=.5, random_state=0)

# Scikit-Learn Linear Regression
regr = skl_lm.LinearRegression()
regr.fit(X_train, y_train)
pred = regr.predict(X_test)
mse = mean_squared_error(y_test, pred)
print(mse)

# Leave One Out Cross Validation
regr = skl_lm.LinearRegression()
loo = LeaveOneOut()
print(loo.get_n_splits(X_poly))
score = cross_val_score(regr, X_poly, auto.mpg, cv=loo, scoring='neg_mean_squared_error').mean()
print(score)

# K Fold Cross Validation
kf = KFold(n_splits=10, random_state=0, shuffle=False)
print(kf.get_n_splits(X_poly))
score = cross_val_score(regr, X_poly, auto.mpg, cv=kf, scoring='neg_mean_squared_error').mean()
print(score)

# BOOTSTRAPPING

def alpha(data, num_samples=100):
    # make a num_samples random choice of indices WITH REPLACEMENT
    indices = np.random.choice(data.index, num_samples, replace=True)
    
    X = data.X[indices].values
    Y = data.Y[indices].values
    
    # np.cov returns full cov matrix we need [0][1] cov(x,y)
    return (np.var(Y) - np.cov(X,Y)[0][1])/(np.var(X) + np.var(Y) - 2*np.cov(X,Y)[0][1])

portfolio = pd.read_csv('portfolio.csv')
print(alpha(portfolio))

def boot(data, statistic_calculator, num_samples = 1000):
    stat_samples = []
    for sample in range(num_samples):
        stat_samples.append(statistic_calculator(data))
    se_estimate = np.std(stat_samples)
    print('Bootstrapped Std. Error(s) =', se_estimate)
    
np.random.seed(0)
print(boot(portfolio, alpha))

auto = pd.read_csv('auto.csv')
auto['horsepower'] = pd.to_numeric(auto.horsepower, errors='coerce')
auto['mpg'] = pd.to_numeric(auto.mpg, errors='coerce')
print(auto.head())

est = smf.ols('mpg ~ horsepower', auto).fit()
print(est.summary().tables[1])
