# perform standard imports
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
PATH = '/home/ec2-user/environment/islr/6_linear_models/figs/'

# Load the Hitters dataset. Pandas offers great flexibillity with dealing with missing values as keywords to 
# pd.read but lets read in all values so we can see how many we are missing.
hitters = pd.read_csv('Hitters.csv', index_col=0)

# Get the number of players and the number of players with missing values
print('Hitters contains', len(hitters), 'players.')
num_missing = np.sum(hitters.isnull().any(axis=1))
print('We are missing data for', num_missing, 'players.')

# now remove the missing players for dataframe
hitters = hitters.dropna()
print('After removal Hitters contains', len(hitters), 'players.')
print('Shape=', hitters.shape)
print(hitters.head())

# Create a set of dummy variables for the categoricals
dummies = pd.get_dummies(hitters[['League', 'Division', 'NewLeague']])
print(dummies.head())

# Generate new dataframe with new dummy variables
df = hitters.drop(['League', 'Division', 'NewLeague'], axis=1)
# add new dummy variables
df = pd.concat([df, dummies[['League_N', 'Division_W', 'NewLeague_N']]],axis=1)
print(df.head(2))

# BEST SUBSET SELECTION

def best_subsets(dataframe, predictors, response, max_features=8):
    """
    Regresses response onto subsets of the predictors in dataframe. Compares models 
    with equal feature numbers choosing the one with the lowest RSS as the 'best' 
    model for that number of features.
    
    PARAMETERS
    ----------
    dataframe : pandas dataframe obj containing responses and predictors
    predictors : list of column names of dataframe used as features
    response : list of column name of dataframe used as target
    
    RETURNS
    -------
    a list of best models, one per feature number
    
    ex.
    [best 1 feat model, best two feat model] = best_subsets(df, predictors, response, max_features = 2)
    """
    
    def process_linear_model(features):
        """
        Constructs Linear Model Regression of response onto features.
        """
        # Create design Matrix
        X = sm.add_constant(dataframe[features])
        y = dataframe[response]

        model = sm.OLS(y,X).fit()
        RSS = model.ssr
        return (model, RSS)

    def get_best_kth_model(k):
        """
        Returns the model from all models with k-predictors with the lowest RSS.
        """
        results = []

        for combo in combinations(predictors, k):
            # process linear model with this combo of features
            results.append(process_linear_model(list(combo)))

        # sort the models and return the one with the smallest RSS
        return sorted(results, key=itemgetter(1)).pop(0)[0]
    
    models =[]
    for k in tqdm(range(1,max_features+1)):
        models.append(get_best_kth_model(k))
    
    return models

# call our best_subsets function
predictors = list(df.columns)
predictors.remove('Salary')
# 19 takes too long
# models = best_subsets(df, predictors, ['Salary'], max_features=19)
models = best_subsets(df, predictors, ['Salary'], max_features=2)
# Output the best set of variables for each model size.
for model in models:
    print(model.model.exog_names)

# We can also look at the R^2 values.
for model in models:
    print(model.rsquared)

# Now that we have the best models for a given number of varaibles we can compare models with different
# predictors using aic, bic and r_adj. Note AIC and Mallow's Cp are proportional to each other. We will 
# create plots of these statistics to find the best model for baseball player salary.
aics = [model.aic for model in models]
bics = [model.bic for model in models]
r_adj = [model.rsquared_adj for model in models]

# find the mins/maxes
min_aic_index, min_aic = min(enumerate(aics), key=itemgetter(1))
min_bic_index, min_bic = min(enumerate(bics), key=itemgetter(1))
max_radj_index, max_radj = max(enumerate(r_adj), key=itemgetter(1))

num_predictors = np.linspace(1, len(models), len(models))
# Create a plot
fig,(ax1, ax2) = plt.subplots(1, 2, figsize=(16,4))

# Plots
ax1.plot(num_predictors, aics, 'r', marker='o', label='AIC');
ax1.plot(num_predictors, bics, 'b', marker='o', label='BIC')

# add the minimums to the axis
ax1.plot(min_aic_index+1, min_aic, 'gx', markersize=20, markeredgewidth=1)
ax1.plot(min_bic_index+1, min_bic, 'gx', markersize=20, markeredgewidth=1)

# Labels and Legend
ax1.set_xlabel('Number of Predictors');
ax1.legend(loc='best');

# Add Adj R**2
ax2.plot(num_predictors, r_adj,'k', marker='o')
ax2.plot(max_radj_index+1, max_radj, 'gx', markersize=20, markeredgewidth=1)
ax2.set_xlabel('Number of Predictors');
ax2.set_ylabel(r'Adjusted $R^2$');

# The Lowest BIC model has the following coeffecients
# print(models[5].params)
print(models[1].params)
plt.savefig(PATH + 'best_subset.png', dpi=300)
plt.close()

# FORWARD AND BACKWARD STEPWISE SELECTION

def forward_step_select(df, predictors, response, max_features=len(predictors)):
    """
    Regresses response onto predictors using a forward step algorithm. Features 
    are added based on minimum RSS.
    
    PARAMETERS
    -----------
    df : dataframe containing predictors and responses
    predictors : list of all possible model predictors
    response : list[variable] to regress onto predictors in df
    max_features : maximum number of features to use from predictors list
    
    RETURNS
    --------
    list of models with increasing number of features upto max_features
    
    """
    
    def process_linear_model(features):
        """
        Constructs Linear Model Regression of response onto features.
        """
        # Create design Matrix
        X = sm.add_constant(df[features])
        y = df[response]

        model = sm.OLS(y,X).fit()
        RSS = model.ssr
        return (model, RSS)

    def update_model(best_features, remaining_features):
        """
        Computes the RSS of possible new models and returns the model with the lowest RSS.
        """
        results = []
        
        for feature in remaining_features:
            results.append(process_linear_model(best_features + [feature]))
            
        # select model with the lowest RSS
        new_model = sorted(results, key= itemgetter(1)).pop(0)[0]
        new_features = list(new_model.params.index)[1:]
        
        return new_features, new_model
    
    # Create list to hold models, model features and the remaining features to test
    models = []
    best_features = []
    remaining_features = predictors
    
    while remaining_features and len(best_features) < max_features:
        
        # get the best new feature set from update_model
        new_features, new_model = update_model(best_features, remaining_features)
        # update the best features to include the one we just found
        best_features = new_features  
        # reduce the available features for the next round
        remaining_features =  [feature for feature in predictors if feature not in best_features]
        
        # append the new_features and model so we can compare models with different features later
        models.append(new_model)
        
    return models

# Call our forward step function
# set up inputs
predictors = list(df.columns)
predictors.remove('Salary')
# call forward_step_select
# 19 too many, takes too long
# mods = forward_step_select(df,predictors,['Salary'],max_features=19)
mods = forward_step_select(df,predictors,['Salary'],max_features=2)
# Output the best set of variables for each model size.
for model in mods:
    print(model.model.exog_names)


def backward_step_select(df, predictors, response):
    
    def process_linear_model(features):
        """
        Constructs Linear Model Regression of response onto features.
        """
        # Create design Matrix
        X = sm.add_constant(df[features])
        y = df[response]

        model = sm.OLS(y,X).fit()
        RSS = model.ssr
        return (model, RSS)

    def update_model(best_features):
        """
        Computes the RSS of possible new models and returns the model with the lowest RSS.
        """
        results = []
        
        for feature in best_features:
            results.append(process_linear_model([x for x in best_features if x != feature]))
            
        # select model with the lowest RSS
        new_model = sorted(results, key= itemgetter(1)).pop(0)[0]
        new_features = list(new_model.params.index)[1:]
        
        return new_features, new_model
    
    models = []
    best_features = predictors
        
    while len(best_features) > 0:
        
        # get the best new feature set from update_model
        best_features, new_model = update_model(best_features)
        
        # append the new_features and model so we can compare models with different features later
        models.append(new_model)
        
    return models

# Call our forward step function
# set up inputs
predictors = list(df.columns)
predictors.remove('Salary')
# call forward_step_select
models_b = backward_step_select(df,predictors,['Salary'])
# Output the best set of variables for each model size.
for model in models_b:
    print(model.model.exog_names)


# Choosing Among Models Using the Validation Set Approach and Cross-Validation

np.random.seed(0)
df_train = df.sample(frac=0.5)
df_test = df.drop(df_train.index)
predictors = list(df_train.columns)
predictors.remove('Salary')
# 19 takes too long
# models = best_subsets(df_train, predictors, ['Salary'], max_features=19)
models = best_subsets(df_train, predictors, ['Salary'], max_features=2)

mses = np.array([])
for model in models:
    # get the predictors for this model, ignore constant
    features = list(model.params.index[1:])
    
    # get the corresponding columns of df_test
    X_test = sm.add_constant(df_test[features])
    
    # make prediction for this model
    salary_pred = model.predict(X_test)
    
    # get the MSE for this model
    mses = np.append(mses, np.mean((salary_pred - df_test.Salary.values)**2))
print('MSEs =', mses)
min_index, min_mse = min(enumerate(mses), key=itemgetter(1))
print(min_index, min_mse)

# set predictors for x-axis
num_predictors = np.linspace(1,len(models),len(models))

fig, ax1 = plt.subplots(figsize=(8,4));

# add the mse and mimimum mse to the plot
ax1.plot(num_predictors, mses, 'r', marker='o', label='MSE')
ax1.plot(min_index+1, min_mse, 'gx', markersize=20, markeredgewidth=2)

# Labels and Legend
ax1.set_xlabel('Number of Predictors');
ax1.set_ylabel('Validation MSE');
ax1.legend(loc='best');
plt.savefig(PATH + 'cross_val.png', dpi=300)
plt.close()

# Now we construct models from the FULL DATASET selecting the best models using our best_subsets function
predictors = list(df.columns)
predictors.remove('Salary')
# 19 takes too long
# models = best_subsets(df, predictors, ['Salary'], max_features=19)
models = best_subsets(df, predictors, ['Salary'], max_features=2)

# Print out the Coeffecients of the 14 predictor model determined as Best by Validation approach above.
# print(models[13].params)
print(models[1].params)

# Create the 10 folds using sklearn KFolds
kf = KFold(n_splits=10, shuffle=True, random_state=1)

# 10Xp matrix to store MSE for each n-variable model for each fold
mses = np.zeros([10, len(predictors)])
fold = 0
for train_index, test_index in kf.split(df):
    # split data for this fold
    df_train = df.iloc[train_index]
    df_test = df.iloc[test_index]
    
    # compute the best model subsets using our function
    models = best_subsets(df_train, predictors, ['Salary'], max_features=2)
    
    # compute the MSE of each model
    for idx, model in enumerate(models):
        # get the predictors for this model, ignore constant
        features = list(model.params.index[1:])
        # get the corresponding columns of df_test
        X_test = sm.add_constant(df_test[features])
        # make prediction for this model
        salary_pred = model.predict(X_test)
        # get the MSE for this model and fold
        mses[fold, idx] = np.mean((salary_pred - df_test.Salary.values)**2)
    fold += 1
        
# now we can compute the mean MSE across folds, one per model with idx features
cvs = np.mean(mses, axis=0)
# We can also plot all the models CV-Errors
# set predictors for x-axis
num_predictors = np.linspace(1,len(models),len(models))

fig, ax1 = plt.subplots(figsize=(8,4));

# get the minimum in the CV
min_index, min_CV = min(enumerate(cvs), key=itemgetter(1))

# add the mse and mimimum mse to the plot
ax1.plot(num_predictors, cvs[0:2], 'b', marker='o', label='Test MSE')        
# ax1.plot(num_predictors, cvs, 'b', marker='o', label='Test MSE')
ax1.plot(min_index+1, min_CV, 'rx', markersize=20, markeredgewidth=2)

# Labels and Legend
ax1.set_xlabel('Number of Predictors');
ax1.set_ylabel('CV Error');
ax1.legend(loc='best');
plt.savefig(PATH + 'errors.png', dpi=300)
plt.close()

# Compute the best subset models
# models = best_subsets(df, predictors, ['Salary'], max_features=19)
models = best_subsets(df, predictors, ['Salary'], max_features=2)
# print the parameters of the 11th model; the model with the lowest Test MSE determined by 10-fold Cross-Validation
print(models[1].params)
