# Standard imports
import warnings

# Math and data processing
import numpy as np
import scipy as sp
import pandas as pd
import patsy as pt

# StatsModels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from patsy import dmatrix

# scikit-learn
from sklearn.preprocessing import PolynomialFeatures

# Visulization
from IPython.display import display
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use('ggplot')
PATH = '/home/ec2-user/environment/islr/7_moving_beyond_linearity/figs/'

# wage dataset is in R ISLR package
wage = pd.read_csv('wage.csv', index_col=0)
print(wage.head(5))
print(wage.info())


def ortho_poly_fit(x, degree = 1):
    '''
    Convert data into orthogonal basis for polynomial regression by QR decomposition.
    Ref: http://davmre.github.io/python/2013/12/15/orthogonal_poly
    '''
    n = degree + 1
    x = np.asarray(x).flatten()
    if(degree >= len(np.unique(x))):
            print("'degree' must be less than number of unique points")
    xbar = np.mean(x)
    x = x - xbar
    X = np.fliplr(np.vander(x, n))
    q,r = np.linalg.qr(X)

    z = np.diag(np.diag(r))
    raw = np.dot(q, z)

    norm2 = np.sum(raw**2, axis=0)
    alpha = (np.sum((raw**2)*np.reshape(x,(-1,1)), axis=0)/norm2 + xbar)[:degree]
    Z = raw / np.sqrt(norm2)
    return Z, norm2, alpha

# Lab 7.8.1 Polynomial Regression and Step Functions
# Polynomial regression of degree 4 on orthogonalized X. Refer to chapter 3 notebook.
X4_ortho = ortho_poly_fit(wage[['age']], degree=4)[0]
X4_ortho[:,0]=1  # Replace constant column with 1s for Intercept estimation.
poly4_ortho = sm.GLS(wage['wage'], X4_ortho).fit()
print(poly4_ortho.summary())

# Polynomial regression of degree 4 on raw X without orthogonalization.
X4 = PolynomialFeatures(degree=4).fit_transform(wage[['age']])
poly4 = sm.GLS(wage['wage'], X4).fit()
print(poly4.summary())

# Predict over a grid of age.
# Generate a sequence of age values spanning the range
age_grid = np.arange(wage.age.min(), wage.age.max()).reshape(-1,1)

# Generate test data
X_test = PolynomialFeatures(4).fit_transform(age_grid)

# Predict the value of the generated ages
y_pred = poly4.predict(X_test)

# Plot
fig, ax = plt.subplots(figsize=(8,6))
fig.suptitle('Degree-4 Polynomial Regression', fontsize=14)

# Scatter plot with polynomial regression line
plt.scatter(wage.age, wage.wage, facecolor='None', edgecolor='k', alpha=0.5)
plt.plot(age_grid, y_pred, color = 'b')
ax.set_ylim(ymin=0)
plt.xlabel('Age')
plt.ylabel('Wage')
plt.savefig(PATH + 'grid_of_age.png', dpi=300)
plt.close()

# ANOVA
X1 = PolynomialFeatures(degree=1).fit_transform(wage[['age']])
X2 = PolynomialFeatures(degree=2).fit_transform(wage[['age']])
X3 = PolynomialFeatures(degree=3).fit_transform(wage[['age']])
X5 = PolynomialFeatures(degree=5).fit_transform(wage[['age']])
poly1 = sm.GLS(wage['wage'], X1).fit()
poly2 = sm.GLS(wage['wage'], X2).fit()
poly3 = sm.GLS(wage['wage'], X3).fit()
poly5 = sm.GLS(wage['wage'], X5).fit()
# ANOVA, as in chpater 3 notebook
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")  ## Supress warnings
    print(anova_lm(poly1, poly2, poly3, poly4, poly5))
    
# Polynomial regression of degree 5 on orthogonalized X. Refer to chapter 3 notebook.
X5_ortho = ortho_poly_fit(wage[['age']], degree=5)[0]
X5_ortho[:,0]=1  # Replace constant column with 1s for Intercept estimation.
poly5_ortho = sm.GLS(wage['wage'], X5_ortho).fit()
print(poly5_ortho.summary())

# Create binary qualitative response
y_clf = (wage.wage > 250).map({False:0, True:1})
# Logistic regression
logreg = sm.GLM(y_clf, X4, family=sm.families.Binomial()).fit()
print(logreg.summary())

# Predict on age grid
y_pred_clf = logreg.predict(X_test)
fig, ax = plt.subplots(figsize=(8, 6))
plt.plot(age_grid, y_pred_clf, color='b')
plt.ylim(-0.02, 0.12)
# Rug plot showing the distribution of wage>250 in the training data.
# 'True' on the top, 'False' on the bottom.
ax.scatter(wage.age, y_clf/10, s=30, c='grey', marker='|', alpha=0.7)
plt.title("Logistic regression with polynomial of degree 4.")
plt.savefig(PATH + 'log_regr.png', dpi=300)
plt.close()

# Step functions for piecewise-constant regression
# Cut predictor data into intervals of a step function
wage_cut, bins = pd.cut(wage.age, 4, retbins=True, right=True)
print(wage_cut.value_counts(sort=False))
# Create dummies for predictor intervals
wage_step_dummies = pd.get_dummies(wage_cut, drop_first=True)  # The first interval is the base and dropped
wage_step_dummies = pd.DataFrame(sm.add_constant(wage_step_dummies.values), columns=['(Intercept)'] + list(wage_step_dummies.columns.values), index=wage_step_dummies.index)
print(wage_step_dummies.head(25))

# Piecewise-constant regression as a step function
logreg_step = sm.GLM(wage.wage, wage_step_dummies).fit()
print(logreg_step.summary())

# SPLINES
# Specifying 3 knots
transformed_3knots = dmatrix("bs(wage.age, knots=(25,40,60), degree=3, include_intercept=False)",
                         {"wage.age": wage.age}, return_type='dataframe')

# Build a regular linear model from the splines
spln_3knots = sm.GLM(wage.wage, transformed_3knots).fit()
pred_3knots = spln_3knots.predict(dmatrix("bs(age_grid, knots=(25,40,60), degree=3, include_intercept=False)",
                                          {"age_grid": age_grid}, return_type='dataframe'))
print(spln_3knots.params)
# Specifying 6 degrees of freedom 
transformed_deg6 = dmatrix("bs(wage.age, df=6, include_intercept=False)",
                        {"wage.age": wage.age}, return_type='dataframe')
spln_deg6 = sm.GLM(wage.wage, transformed_deg6).fit()
pred_deg6 = spln_deg6.predict(dmatrix("bs(age_grid, df=6, degree=3, include_intercept=False)",
                                  {"age_grid": age_grid}, return_type='dataframe'))
print(spln_deg6.params)

# Natural splines
# Specifying 4 degrees of freedom
transformed_deg4 = dmatrix("cr(wage.age, df=4)", {"wage.age": wage.age}, return_type='dataframe')
spln_deg4 = sm.GLM(wage.wage, transformed_deg4).fit()
pred_deg4 = spln_deg4.predict(dmatrix("cr(age_grid, df=4)", {"age_grid": age_grid}, return_type='dataframe'))
print(spln_deg4.params)
# Plot splines
fig, ax = plt.subplots(figsize=(8, 6))
plt.scatter(wage.age, wage.wage, facecolor='None', edgecolor='k', alpha=0.3)
plt.plot(age_grid, pred_3knots, color='b', label='Specifying three knots')
plt.plot(age_grid, pred_deg6, color='r', label='Specifying df=6')
plt.plot(age_grid, pred_deg4, color='g', label='Natural spline df=4')
[plt.vlines(i , 0, 350, linestyles='dashed', lw=2, colors='b') for i in [25,40,60]]
plt.legend(bbox_to_anchor=(1.5, 1.0))
plt.xlim(15,85)
plt.ylim(0,350)
plt.xlabel('age')
plt.ylabel('wage')
plt.savefig(PATH + 'splines.png', dpi=300)
plt.close()

wage['education'] = wage['education'].map({'1. < HS Grad': 1.0, 
                                                 '2. HS Grad': 2.0, 
                                                 '3. Some College': 3.0,
                                                 '4. College Grad': 4.0,
                                                 '5. Advanced Degree': 5.0
                                                })

# Use patsy to generate entire matrix of basis functions
X = pt.dmatrix('cr(year, df=4)+cr(age, df=5) + education', wage)
y = np.asarray(wage['wage'])

def confidence_interval(X, y, y_hat):
    """Compute 5% confidence interval for linear regression"""
    # STATS
    # ----------------------------------
    # Reference: https://stats.stackexchange.com/questions/44838/how-are-the-standard-errors-of-coefficients-calculated-in-a-regression
    
    # Covariance of coefficient estimates
    mse = np.sum(np.square(y_hat - y)) / y.size
    cov = mse * np.linalg.inv(X.T @ X)
    # ...or alternatively this stat is provided by stats models:
    #cov = model.cov_params()
    
    # Calculate variance of f(x)
    var_f = np.diagonal((X @ cov) @ X.T)
    # Derive standard error of f(x) from variance
    se       = np.sqrt(var_f)
    conf_int = 2*se
    return conf_int

# GAM

# Fit logistic regression model
model = sm.OLS(y, X).fit(disp=0)
y_hat = model.predict(X)
conf_int = confidence_interval(X, y, y_hat)
# Plot estimated f(year)
sns.lineplot(x=wage['year'], y=y_hat)
plt.savefig(PATH + 'wage_year.png', dpi=300)
plt.close()

# Plot estimated f(age)
sns.lineplot(x=wage['age'], y=y_hat)
plt.savefig(PATH + 'wage_age.png', dpi=300)
plt.close()

# Plot estimated f(education)
# sns.boxplot(x=wage['education'], y=y_hat)
sns.lineplot(x=wage['education'], y=y_hat)
plt.savefig(PATH + 'wage_educ.png', dpi=300)
plt.close()
# Not quite the same as plots achived by ISL authors using R, but gives similar insight.

# Comparing GAM configurations with ANOVA
# Model 1
X = pt.dmatrix('cr(age, df=5) + education', wage)
y = np.asarray(wage['wage'])
model1 = sm.OLS(y, X).fit(disp=0)

# Model 2
X = pt.dmatrix('year+cr(age, df=5) + education', wage)
y = np.asarray(wage['wage'])
model2 = sm.OLS(y, X).fit(disp=0)
# Model 3
X = pt.dmatrix('cr(year, df=4)+cr(age, df=5) + education', wage)
y = np.asarray(wage['wage'])
model3 = sm.OLS(y, X).fit(disp=0)

# Compare models with ANOVA
print(sm.stats.anova_lm(model1, model2, model3))

# We condlude that inclusion of a linear year feature improves the model, but there
# is no evidence that a non-linear function of year improves the model.
print(model3.summary())