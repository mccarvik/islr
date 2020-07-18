import pdb
import numpy as np
import pandas as pd
import patsy as pt
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn import svm
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import GridSearchCV
PATH = '/home/ec2-user/environment/islr/9_support_vector_machines/figs/'

# Generate multi-class data with 3 response classes 0, 1 and 2
np.random.seed(0)
X = np.random.normal(0, 1, (300, 2))
X[1:100,]   = X[1:100,]
X[101:200, :1] = X[101:200, :1] + 4
X[201:300, 1:] = X[201:300, 1:] + 4
X[201:300, :1] = X[201:300, :1] + 2
y = np.concatenate((np.repeat(0, 100), np.repeat(1, 100), np.repeat(2, 100)))
# Plot data
df = pd.concat([pd.DataFrame(data=X, columns=['x1', 'x2']), pd.Series(y, name='y')], axis=1)
plt.figure(figsize=(10, 10))
sns.scatterplot(x='x1', y='x2', hue='y', data=df);
plt.savefig(PATH + 'mult_classes.png', dpi=300)
plt.close()

model = svm.SVC(kernel='rbf', gamma=1, C=1, random_state=0, probability=True).fit(X, y)
# Decision boundary plot
x1 = np.arange(-5, 7, .1)
x2 = np.arange(-5, 7, .1)
xx1, xx2 = np.meshgrid(x1, x2, sparse=False)

Xgrid = np.stack((xx1.flatten(), xx2.flatten())).T
y_hat = model.predict(Xgrid)
y_grid = y_hat.reshape(len(x2), len(x1))

fig = plt.figure(figsize=(10, 10))
plt.contour(x1, x2, y_grid);

sns.scatterplot(x='x1', y='x2', hue='y', data=df)
#sns.scatterplot(x=model.support_vectors_[:,0], y=model.support_vectors_[:,1], color='red', marker='+', s=500)
plt.savefig(PATH + 'mult_scatter.png', dpi=300)
plt.close()

pdb.set_trace()
X_train = pd.read_csv('khan_xtrain.csv', index_col=0)
X_test = pd.read_csv('khan_xtest.csv', index_col=0)
y_train = pd.read_csv('khan_ytrain.csv', index_col=0)
y_test = pd.read_csv('khan_ytest.csv', index_col=0)
svc = SVC(kernel='linear', C=10)
svc.fit(X_train, y_train)

# Training set Confusion Matrix and Performance¶
pred_train = svc.predict(X_train)
print("Training Accuracy: ", metrics.accuracy_score(y_train, pred_train) )
print("Training Sensitivity: ",  metrics.recall_score(y_train, pred_train, average='weighted') )
print("Training Precision: ",  metrics.precision_score(y_train, pred_train, average='weighted') )
conf_mat = metrics.confusion_matrix(y_train, pred_train) ; conf_mat

# Test set Confusion Matrix and Performance
pred_test = svc.predict(X_test)
conf_mat = metrics.confusion_matrix(y_test, pred_test) 
conf_mat_df = pd.DataFrame(conf_mat, index=svc.classes_, columns=svc.classes_)
conf_mat_df.index.name = "True(실제)"
conf_mat_df.columns.name = "Predicted"
print(conf_mat_df)
print()
print("Test Accuracy: ", metrics.accuracy_score(y_test, pred_test) )
print("Test Sensitivity: ",  metrics.recall_score(y_test, pred_test, average='weighted') )
print("Test Precision: ",  metrics.precision_score(y_test, pred_test, average='weighted') )
parameters = [{'C': [0.01, 0.1, 0.5, 1, 5, 10, 100, 1000]}]
svm_multi_clf_grid = GridSearchCV(SVC(kernel='linear'), parameters, cv=8, scoring='accuracy')
svm_multi_clf_grid.fit(X_train, y_train)
print(svm_multi_clf_grid.cv_results_ )

# accuracy
print(svm_multi_clf_grid.best_score_)
print(svm_multi_clf_grid.best_params_)
svc_best = SVC(kernel='linear', C=0.01)
svc_best.fit(X_train, y_train)

pred_test = svc_best.predict(X_test)
conf_mat = metrics.confusion_matrix(y_test, pred_test) 
conf_mat_df = pd.DataFrame(conf_mat, index=svc.classes_, columns=svc.classes_)
conf_mat_df.index.name = "True(실제)"
conf_mat_df.columns.name = "Predicted"
print(conf_mat_df)
print()
print("Test Accuracy: ", metrics.accuracy_score(y_test, pred_test) )
print("Test Sensitivity: ",  metrics.recall_score(y_test, pred_test, average='weighted') )
print("Test Precision: ",  metrics.precision_score(y_test, pred_test, average='weighted') )
