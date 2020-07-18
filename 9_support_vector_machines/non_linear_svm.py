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

# Generate data with a non-linear class boundary
np.random.seed(1)
X = np.random.normal(0, 1, (200, 2))
X[1:100,]   = X[1:100,] + 2
X[101:150,] = X[101:150,] - 2
y = np.concatenate((np.repeat(0, 150), np.repeat(1, 50)))

# Plot data
df = pd.concat([pd.DataFrame(data=X, columns=['x1', 'x2']), pd.Series(y, name='y')], axis=1)
plt.figure(figsize=(10, 10))
sns.scatterplot(x='x1', y='x2', hue='y', data=df);
plt.savefig(PATH + 'nonlinear.png', dpi=300)
plt.close()

# Index a training set
train = np.random.random(len(y)) > 0.5
model = svm.SVC(kernel='rbf', gamma=1, C=1, random_state=0).fit(X[train], y[train])

# Decision boundary plot
x1 = np.arange(-5, 5, .1)
x2 = np.arange(-5, 5, .1)
xx1, xx2 = np.meshgrid(x1, x2, sparse=False)

Xgrid = np.stack((xx1.flatten(), xx2.flatten())).T
y_hat = model.predict(Xgrid)
y_grid = y_hat.reshape(len(x2), len(x1))
y_grid.shape

fig = plt.figure(figsize=(10, 10))
plt.contourf(x1, x2, y_grid);
sns.scatterplot(x='x1', y='x2', hue='y', data=df[train])
sns.scatterplot(x=model.support_vectors_[:,0], y=model.support_vectors_[:,1], color='red', marker='+', s=500)
plt.savefig(PATH + 'nonlinear_fit.png', dpi=300)
plt.close()

# Get summary of model
print('Model parameters:')
print(model.get_params)

print('\nNumber of support vectors for each class.:')
print(model.n_support_)

print('\nCoefficients of the support vector in the decision function. :')
print(model.dual_coef_)

print('\nTraining accuracy:')
print(model.score(X[train], y[train]))

# SMALLER COST
model = svm.SVC(kernel='rbf', gamma=1, C=1e5, random_state=0).fit(X[train], y[train])
# Decision boundary plot

x1 = np.arange(-5, 5, .1)
x2 = np.arange(-5, 5, .1)
xx1, xx2 = np.meshgrid(x1, x2, sparse=False)

Xgrid = np.stack((xx1.flatten(), xx2.flatten())).T
y_hat = model.predict(Xgrid)
y_grid = y_hat.reshape(len(x2), len(x1))
y_grid.shape

fig = plt.figure(figsize=(10, 10))
plt.contourf(x1, x2, y_grid);
sns.scatterplot(x='x1', y='x2', hue='y', data=df[train])
sns.scatterplot(x=model.support_vectors_[:,0], y=model.support_vectors_[:,1], color='red', marker='+', s=500)
plt.savefig(PATH + 'nonlinear_fit_small_cost.png', dpi=300)
plt.close()

# Get summary of model
print('Model parameters:')
print(model.get_params)

print('\nNumber of support vectors for each class.:')
print(model.n_support_)

print('\nCoefficients of the support vector in the decision function. :')
print(model.dual_coef_)

print('\nTraining accuracy:')
print(model.score(X[train], y[train]))

# HYPERPARAMATER TRAINING
# Train Classifiers
# https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html#sphx-glr-auto-examples-svm-plot-rbf-parameters-py
# ----------------------------------------------------------------
C_range     = np.logspace(-2, 10, 3)
gamma_range = np.logspace(-3, 9, 3) 
param_grid  = dict(gamma=gamma_range, C=C_range)
grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=5)
grid.fit(X[train], y[train])
print(f"The best parameters are {grid.best_params_}, with a score of {grid.best_score_:.3f}")


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Stolen from here: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.title("Normalized confusion matrix")
    else:
        plt.title('Confusion matrix, without normalization')
        
    #plt.title(title)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.grid(False)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(PATH + 'nonlinear_confusion.png', dpi=300)
    plt.close()
    

cm = confusion_matrix(y[~train], grid.predict(X[~train]))
plot_confusion_matrix(cm, ['True', 'False'], normalize=False)
print(f"{(9 / len(y[~train]))*100:.2f}% of test observations are misclassified by this SVM.")

# 9.6.3 ROC Curves
# A single ROC curve
# Fit optimal model chosen by grid search 
model = svm.SVC(kernel='rbf', gamma=1, C=1, random_state=0, probability=True).fit(X[train], y[train])
# Get probability of positive binary classification
probs = model.predict_proba(X[~train])
preds = probs[:, 1]
# Get ROC metrics
# False Postitive Rate, True Positive Rate metrics by threshold
fpr, tpr, threshold = metrics.roc_curve(y[~train], preds)
# Get area under curve metrics
auc = metrics.auc(fpr, tpr)
# Plot ROC curve using seaborn
plot_df = pd.DataFrame({'False Positive Rate': fpr, 'True Positive Rate': tpr})
plt.figure(figsize=(10,10))
sns.lineplot(x='False Positive Rate', y='True Positive Rate', data=plot_df, 
             estimator=None, 
             label=f'ROC curve (area={auc:.3f})');
plt.savefig(PATH + 'roc_curve.png', dpi=300)
plt.close()

# ROC comparison
results = np.empty((0, 3))
for g in np.logspace(-5, 4, base=2, num=4):
    # Fit optimal model chosen by grid search 
    model = svm.SVC(kernel='rbf', gamma=g, C=1, random_state=0, probability=True).fit(X[train], y[train])
    # Get probability of positive binary classification
    probs = model.predict_proba(X[~train])
    preds = probs[:, 1]
    # Get ROC metrics
    # False Postitive Rate, True Positive Rate metrics by threshold
    fpr, tpr, threshold = metrics.roc_curve(y[~train], preds)
    # Get area under curve metrics
    auc = metrics.auc(fpr, tpr)
    r = np.array([np.repeat(g, len(fpr)), fpr, tpr]).T
    results = np.concatenate((results, r), axis=0)
plot_df = pd.DataFrame(results, columns=['gamma', 'fpr', 'tpr'])
plt.figure(figsize=(10,10))
sns.lineplot(x='fpr', y='tpr', hue='gamma', data=plot_df, estimator=None)
plt.savefig(PATH + 'roc_comparison.png', dpi=300)
plt.close()
