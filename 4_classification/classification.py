## perform imports and set-up
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
PATH = '/home/ec2-user/environment/islr/4_classification/figs/'


# print with precision 4
pd.set_option('precision', 4)

market_df = pd.read_csv('smarket.csv')
market_df = market_df.rename(columns = {'Unnamed: 0':'Day'})
print(market_df.head())
print(market_df.shape)
print(market_df.describe())
print(market_df.corr(method='pearson'))

fig, ax = plt.subplots(figsize = (8,6))
ax.scatter(market_df.Day, market_df.Volume);
ax.set_xlabel('Days');
ax.set_ylabel('Volume in Billions');
plt.savefig(PATH + 'vol.png', dpi=300)
plt.close()

# LOGISTIC REGRSSION
# Encode the response as 0,1 for down/up
market_df['DirCoded'] = [0 if d == 'Down' else 1 for d in market_df.Direction]
print(market_df.head())

logit_fit = smf.logit('DirCoded ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume', market_df).fit()
print(logit_fit.summary())
print(logit_fit.params)

logit_probs = logit_fit.predict(market_df)
print(logit_probs[:10])
logit_pred = pd.Series(['Up' if p > 0.5 else 'Down' for p in logit_probs])
print(logit_pred[:10])
table = logit_fit.pred_table(threshold=0.5)
confusion_df = pd.DataFrame(table, ['Down','Up'], ['Down','Up'])
print(confusion_df)

print('The model made {} correct predictions on the TRAINING SET.'.format((confusion_df.Down[0] + confusion_df.Up[1])/logit_fit.nobs))
print(np.mean(logit_pred == market_df.Direction))
train_df = market_df[market_df.Year < 2005]
test_df = market_df[market_df.Year == 2005]
print(test_df.shape)

train_fit = smf.logit('DirCoded ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume', train_df).fit()
predicted_probs = train_fit.predict(test_df)

test_df['preds'] = ['Up' if p > 0.5 else 'Down' for p in predicted_probs]
print('The model made', np.mean(test_df.preds == test_df.Direction),'% correct predictions on the TEST SET.')

# Stats models doesn't have an option for a confusion matrix for test sets-- we build one by hand
print(pd.crosstab(test_df.preds, test_df.Direction))
print('The model made', np.mean(test_df.preds != test_df.Direction),'% incorrect predictions on the TEST SET.')

# using just last two days lags
train_fit = smf.logit('DirCoded ~ Lag1 + Lag2', train_df).fit()
predicted_probs = train_fit.predict(test_df)
test_df['preds'] = ['Up' if p > 0.5 else 'Down' for p in predicted_probs]
print('The model made', np.mean(test_df.preds == test_df.Direction),'% correct predictions on the TEST SET.')

# Stats models doesn't have an option for a confusion matrix for test sets-- we build one by hand
table = pd.crosstab(test_df.preds, test_df.Direction)
print(table)
print('On days where the prediction day is \'Up\' the probability the market will be \'Up\' is',
      table['Up']['Up']/(table['Down']['Up'] + table['Up']['Up']))
      
# predict for specific lag vales
print(train_fit.predict(pd.DataFrame({'Lag1':[1.2, 1.5], 'Lag2':[1.1,-0.8]})))

# LINEAR DISCRIMINANT ANALYSIS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# use lag 1 and lag 2 as the predictors
predictors = ['Lag1','Lag2']

X_train = train_df[predictors]
Y_train = train_df.DirCoded

# Create Classifier Instance
lda_clf = LDA()
# Fit model
lda_clf.fit(X_train,Y_train)

print('Class Priors (P(Y = k)) =', lda_clf.priors_)
print('Class Means μk\n   Down:', lda_clf.means_[0], '\n   Up:  ', lda_clf.means_[1])
print('Coeffecients =\n', lda_clf.scalings_)

# Scatter plot the data colored by market direction #
#####################################################
fig, ax = plt.subplots(figsize=(8,8))
# Plot the training lags color coded by market direction
ax.scatter(X_train[Y_train==1].Lag1, X_train[Y_train==1].Lag2, alpha=0.7, label='Up');
ax.scatter(X_train[Y_train==0].Lag1, X_train[Y_train==0].Lag2, alpha=0.7, label='Down');

# Calculate Bayes Decision Boundary #
#####################################
# Construct a meshgrid to calulate Bayes Boundary over
nx, ny = 200, 200
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()
xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))

# Use predict_proba to calculate Probability at each x1,x2 pair
Z = lda_clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z = Z[:, 1].reshape(xx.shape)
# The Bayes Boundary for k=2 classes is the contour where P(Y=k|X=x) = 0.5
cp = ax.contour(xx, yy, Z, [0.5], linewidths=1., colors='k');
plt.clabel(cp, inline=True, fmt='Bayes Decision Boundary',fontsize=8)

# Plot mean lag vector (lag1,lag2) for each class 'Up' and 'Down' #
###################################################################
ax.plot(lda_clf.means_[0][0], lda_clf.means_[0][1], 'o', color='red', markersize=10);
ax.plot(lda_clf.means_[1][0], lda_clf.means_[1][1], 'o', color='blue', markersize=10);

ax.set_xlabel('Lag1')
ax.set_ylabel('Lag2')
ax.legend(loc='best');
plt.savefig(PATH + 'bayes_boundaries.png', dpi=300)
plt.close()
lda_pred_class_coded = lda_clf.predict(test_df[predictors])
print(lda_pred_class_coded)

lda_pred_class = ['Up' if c == 1 else 'Down' for c in lda_pred_class_coded]
test_df['lda_pred_class'] = lda_pred_class
print(lda_pred_class[:10])
# Compute Test Confusion Matrix #
#################################
table = pd.crosstab(test_df.lda_pred_class, test_df.Direction)
print(table)
print(np.mean(test_df.lda_pred_class == test_df.Direction))
lda_pred_posterior = lda_clf.predict_proba(test_df[predictors])
print(lda_pred_posterior)

print(np.sum(lda_pred_posterior[:,0] >= 0.5))
print(np.sum(lda_pred_posterior[:,1] >= 0.5))
print(lda_pred_posterior[:20,0])
print(test_df.lda_pred_class[:20])
print(np.sum(lda_pred_posterior[:,0] >= 0.9))
print(np.max(lda_pred_posterior[:,0]))

# QUADRATIC DISCRIMINANT ANALYSIS
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
# Create Classifier Instance
qda_clf = QDA()
# Fit model
qda_clf.fit(X_train,Y_train)
print('Class Priors (P(Y = k)) =', qda_clf.priors_)
print('Class Means μk\n   Down:', qda_clf.means_[0], '\n   Up:  ', lda_clf.means_[1])

qda_pred_class_coded = qda_clf.predict(test_df[predictors])
qda_pred_class = ['Up' if c == 1 else 'Down' for c in qda_pred_class_coded]
test_df['qda_pred_class'] = qda_pred_class
print('The model makes {0:.4f} correct predictions'.format(100*np.mean(test_df.qda_pred_class == test_df.Direction)))

# Compute Test Confusion Matrix #
#################################
table = pd.crosstab(test_df.qda_pred_class, test_df.Direction)
print(table)

# Scatter plot the data colored by market direction #
#####################################################
fig, ax = plt.subplots(figsize=(8,8))
# Plot the training lags color coded by market direction
ax.scatter(X_train[Y_train==1].Lag1, X_train[Y_train==1].Lag2, alpha=0.7, label='Up');
ax.scatter(X_train[Y_train==0].Lag1, X_train[Y_train==0].Lag2, alpha=0.7, label='Down');

# Calculate Bayes Decision Boundary #
#####################################
# Construct a meshgrid to calulate Bayes Boundary over
nx, ny = 200, 200
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()
xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))

# Use predict_proba to calculate Probability at each x1,x2 pair
Z = qda_clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z = Z[:, 1].reshape(xx.shape)
# The Bayes Boundary for k=2 classes is the contour where P(Y=k|X=x) = 0.5
cp = ax.contour(xx, yy, Z, [0.5], linewidths=1., colors='k');
plt.clabel(cp, inline=True, fmt='Bayes Decision Boundary',fontsize=8)

ax.set_xlabel('Lag1')
ax.set_ylabel('Lag2')
ax.legend(loc='best');
plt.savefig(PATH + 'qda.png', dpi=300)
plt.close()

# K NEAREST NEIGHBORS
from sklearn.neighbors import KNeighborsClassifier as KNNC
# Build a KNN classifier
knn_1 = KNNC(n_neighbors=1)
knn_1.fit(X_train, train_df.Direction)

knn1_pred = knn_1.predict(test_df[predictors])
print(knn1_pred)
print('The model makes {0:.4f}% correct predictions'.format(100*np.mean(knn1_pred == test_df.Direction)))

# Compute Test Confusion Matrix #
#################################
table = pd.crosstab(knn1_pred, test_df.Direction)
print(table)

# use 3 neighbors now
knn_3 = KNNC(n_neighbors=3)
knn_3.fit(X_train, train_df.Direction)
knn3_pred = knn_3.predict(test_df[predictors])
print('The model makes {0:.4f}% correct predictions'.format(100*np.mean(knn3_pred == test_df.Direction)))

# Compute Test Confusion Matrix #
#################################
table = pd.crosstab(knn3_pred, test_df.Direction)
print(table)

# CARAVAN APPLICATION
# Load the caravan insurance data for section 4.6.6
caravan_df = pd.read_csv('caravan.csv', index_col=0)
(caravan_df.head())
print(caravan_df.shape)
print(caravan_df.Purchase.describe())
print('The probability an individual purchased car insurance was,',
      1- caravan_df.Purchase.describe().freq / len(caravan_df.Purchase))
      
# Standardize the Variables #
#############################
X = scale(caravan_df.iloc[:,0:85], axis=0)

# check that X[:,1] is now standardized
print('mean =', np.mean(X[:,0]), 'variance = ', np.var(X[:,0]))
Y = caravan_df.Purchase.values

X_train = X[1000:]
X_test = X[:1000]
Y_train = Y[1000:]
Y_test = Y[:1000]
knn1 = KNNC(n_neighbors=1)
knn1.fit(X_train, Y_train)
knn1_pred = knn1.predict(X_test)
error_rate = np.mean(knn1_pred != Y_test)

print('The error rate for k={0:d} is {1:.3f} %'.format(1, 100*error_rate))
print('There are {} % custumors who bought insurance on the test set.'.format(np.mean(Y_test != 'No')))
table = pd.crosstab(knn1_pred, Y_test)
print(table)
print("The true positive rate on the test set: {}".format(table.values[1,1]/(table.values[1,1] + table.values[1,0])))

# Build a k=3 Neighbor Classifier #
###################################
knn3 = KNNC(n_neighbors=3)
knn3.fit(X_train, Y_train)
knn3_pred = knn3.predict(X_test)
error_rate = np.mean(knn3_pred != Y_test)
print('The error rate for k=3 is {} %'.format(100*error_rate))
table = pd.crosstab(knn3_pred, Y_test)
print("The true positive rate on the test for k=3 set: {}".format(table.values[1,1]/(table.values[1,1] + table.values[1,0])))

# Build a k=5 Neighbor Classifier #
###################################
knn5 = KNNC(n_neighbors=5)
knn5.fit(X_train, Y_train)
knn5_pred = knn5.predict(X_test)
error_rate = np.mean(knn5_pred != Y_test)
print('The error rate for k=5 is {} %'.format(100*error_rate))
table = pd.crosstab(knn5_pred, Y_test)
print("The true positive rate on the test for k=5 set: {}".format(table.values[1,1]/(table.values[1,1] + table.values[1,0])))

# Logistic Regression Approach
predictors = caravan_df.columns.tolist()[:-1]
#formula = 'ResCoded ~ ' + ' + '.join(predictors)
Y_train_coded = [0 if d == 'No' else 1 for d in Y_train]
X_train2 = sm.add_constant(X_train)
logit_fit = sm.Logit(Y_train_coded, X_train2).fit()
X_test2 = sm.add_constant(X_test, has_constant='add')
logit_probs = logit_fit.predict(X_test2)
logit_pred = pd.Series(['Yes' if p > 0.5 else 'No' for p in logit_probs])
table = pd.crosstab(logit_pred, Y_test)
print(table)

# move the threshold from 0.5 to 0.25
logit_pred = pd.Series(['Yes' if p > 0.25 else 'No' for p in logit_probs])
table = pd.crosstab(logit_pred, Y_test)
print(table)
