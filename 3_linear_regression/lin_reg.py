import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from scipy import stats
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston # boston data set is part of sklearn
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from mpl_toolkits.mplot3d import Axes3D
PATH = '/home/ec2-user/environment/islr/3_linear_regression/figs/'


# Load Boston housing data set
boston = load_boston()

#Transform the data into a dataframe for analysis¶

# combine the predictors and responses for a dataframe
predictors = boston.data
response = boston.target
boston_data = np.column_stack([predictors,response])

# now get the column names of the data frame
col_names = np.append(boston.feature_names, 'MEDV')

# create the data frame
boston_df = pd.DataFrame(boston_data, columns = col_names)
print(boston_df.head())
print(boston_df.columns)

lm_fit = smf.ols('MEDV~LSTAT', boston_df).fit()
print(lm_fit.summary())
print(lm_fit.params)
print(lm_fit.conf_int(alpha=0.05))
# print(lm_fit.fittedvalues)

# The statsmodels module can be used to produce confidence intervals and prediction 
# intervals for the prediction of medv for a given value of lstat (predictions).
predictors = pd.DataFrame({'LSTAT':[5,10,15]})
predictions = lm_fit.get_prediction(predictors)
print(predictions.summary_frame(alpha=0.05))

#  Create a plot to plot the data, OLS estimate, prediction and confidence intervals
fig, ax = plt.subplots(figsize=(8,6))

# get numpy array values from dataframe
x = boston_df.LSTAT

# Plot the data
ax.scatter(x, boston_df.MEDV, facecolors='none', edgecolors='b', label="data")
# plot the models fitted values
ax.plot(x, lm_fit.fittedvalues, 'g', label="OLS")

# To plot prediction and confidence intrvals we need predictions for all data points
predictions = lm_fit.get_prediction(boston_df).summary_frame(alpha=0.05)

# plot the high and low prediction intervals
ax.plot(x, predictions.obs_ci_lower, color='0.75', label="Prediction Interval")
ax.plot(x, predictions.obs_ci_upper, color='0.75', label="")

# plot the high and low mean confidence intervals
ax.plot(x, predictions.mean_ci_lower, color='r',label="Predicted Mean CI")
ax.plot(x, predictions.mean_ci_upper, color='r', label="")

ax.legend(loc='best');
plt.xlabel('LSTAT');
plt.ylabel('MEDV');
plt.savefig(PATH + 'medv.png', dpi=300)
plt.close()

# We need this for leverage and studentized residuilas calculations.
from statsmodels.stats.outliers_influence import OLSInfluence
influence = OLSInfluence(lm_fit)
leverage = influence.hat_matrix_diag
stud_res = influence.resid_studentized_external

# Create plots of residuals
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,6))

# Plot the residual for each fitted value
ax1.scatter(lm_fit.fittedvalues, lm_fit.resid, facecolors='none', edgecolors='b');
ax1.set_xlabel('fitted values');
ax1.set_ylabel('residuals');
# The residual plot indicates significant nonlinearity (a u-shape pattern is clear)

# Plot the studentized residuals
ax2.scatter(lm_fit.fittedvalues, stud_res, facecolors='none', edgecolors='b');
ax2.set_ylabel('Studentized Residuals');
ax2.set_xlabel('fitted values');

# |studentized residual| > 3 are generally considered outliers
plt.savefig(PATH + 'dist.png', dpi=300)
plt.close()


# We can also examine the leverages to identify points that may alter the regression line
fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(leverage, stud_res,facecolors='none', edgecolors='b');
ax.set_xlabel('Leverage');
ax.set_ylabel('Studentized Residuals');
plt.savefig(PATH + 'leverages.png', dpi=300)
plt.close()

from scipy.stats import probplot
probplot(lm_fit.resid, plot=plt)
plt.savefig(PATH + 'normal_prob.png', dpi=300)
plt.close()

# Multiple Regression
lm_fit = smf.ols('MEDV~LSTAT+AGE', boston_df).fit()
print(lm_fit.summary())
preds = list(boston_df.columns)
preds.remove('MEDV')
my_formula = 'MEDV~' + '+'.join(preds)
print(my_formula)
lm_ins = smf.ols(my_formula, boston_df)
lm_fit = lm_ins.fit()
print(lm_fit.summary())

# This gives us the R^2
print(lm_fit.rsquared)
# This gives us the RSE
print(np.sqrt(lm_fit.mse_resid))

from statsmodels.stats.outliers_influence import variance_inflation_factor

# exog is the predictor matrix of the model
VIFs = [(predictor, variance_inflation_factor(lm_ins.exog, idx)) 
        for (idx, predictor) in enumerate(lm_ins.exog_names)]

print('Variance Inflation Factors')
for tup in VIFs:
    print('{:10}'.format(tup[0]), '{:.3f}'.format(tup[1]))
    
preds = list(boston_df.columns)
preds.remove('MEDV')
preds.remove('AGE')
my_formula = 'MEDV~' + '+'.join(preds)
print(my_formula)
lm_ins1 = smf.ols(my_formula, boston_df)
lm_fit1 = lm_ins1.fit()
print(lm_fit1.summary())

# Interaction Terms
lm_fit2 = smf.ols('MEDV ~ LSTAT + I(LSTAT**2)', boston_df).fit()
print(lm_fit2.summary())

# Non linear trnasformations
# import anova function
from statsmodels.stats.api import anova_lm
lm_fit = smf.ols('MEDV ~ LSTAT', boston_df).fit()
lm_fit2 = smf.ols('MEDV ~ LSTAT + I(LSTAT**2)', boston_df).fit()
# perform the hypothesis test (see https://en.wikipedia.org/wiki/F-test regression section)
print(anova_lm(lm_fit, lm_fit2))

fig, ax = plt.subplots(figsize=(8,6))

# Plot the data
ax.scatter(boston_df.LSTAT, boston_df.MEDV, facecolors='none', edgecolors='b', label="data");
# plot the models fitted values
ax.plot(boston_df.LSTAT, lm_fit2.fittedvalues, 'g', marker='o',linestyle='none', label="OLS");

ax.legend(loc='best');
plt.xlabel('LSTAT');
plt.ylabel('MEDV');
plt.savefig(PATH + 'quadratic.png', dpi=300)
plt.close()

# Create plots of residuals
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,6))

# Plot the residual for each fitted value for the linear model
ax1.scatter(lm_fit.fittedvalues, lm_fit.resid, facecolors='none', edgecolors='b');
ax1.set_xlabel('fitted values');
ax1.set_ylabel('residuals');
ax1.set_title('Linear Model Residuals')

ax2.scatter(lm_fit2.fittedvalues, lm_fit2.resid, facecolors='none', edgecolors='b');
ax2.set_title('Quadratic Model Residuals');
plt.savefig(PATH + 'resid.png', dpi=300)
plt.close()

# cubic fit
formula = 'MEDV ~ LSTAT +' + ' + '.join('I(LSTAT**{})'.format(i) for i in range(2, 6))
print(formula)
# adding terms up to 5th order improves the model
lm_fit5 = smf.ols(formula, boston_df).fit()
print(lm_fit5.summary())
# log transformation
print(smf.ols('MEDV ~ np.log(LSTAT)', boston_df).fit().summary())

# Qualitative Predictors
carseats_df = pd.read_csv('carseats.csv', index_col = 0)
print(carseats_df.head())
# Construct the formula with two interaction terms
preds = carseats_df.columns.tolist()[1:]
formula ='Sales ~ ' + ' + '.join(preds) + ' + Income:Advertising + Price:Age'
print(formula)
lm_fit = smf.ols(formula, carseats_df).fit()
print(lm_fit.summary())