# Standard imports
import warnings
import pdb

# Math and data processing
import numpy as np
import scipy as sp
import pandas as pd

# scikit-learn
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import scale
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix, classification_report

# Visulization
from IPython.display import display
from IPython.display import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
mpl.style.use('ggplot')
import pydotplus

PATH = '/home/ec2-user/environment/islr/8_decision_trees/figs/'
carseats = pd.read_csv('carseats.csv', index_col=0)
print(carseats.head(5))

# Coding qualitative variables
carseats['High'] = (carseats.Sales > 8).astype('int')
carseats['ShelveLoc'] = carseats.ShelveLoc.map({'Bad':0, 'Good':1, 'Medium':2})
carseats['Urban'] = (carseats.Urban == 'Yes').astype('int')
carseats['US'] = (carseats.US == 'Yes').astype('int')
print(carseats.head())

# Use max_depth to limit tree size, since manual pruning is not implemented in scikit-learn.
X = carseats.drop(['Sales', 'High'], axis=1)
y = carseats['High']
tree = DecisionTreeClassifier(max_depth=6)
tree.fit(X, y)
score = tree.score(X, y)
print("Training error rate = ", 1-score)
print("Tree node size =", tree.tree_.node_count)
# Visulization
dot_data = export_graphviz(tree, out_file=None, feature_names=X.columns, filled=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png("figs/dtree.png")
# Image(graph.create_png(), width=2000)

# Feature Importance
Importance = pd.DataFrame({'Importance':tree.feature_importances_*100}, index=X.columns)
Importance.sort_values('Importance', axis=0, ascending=True).plot(kind='barh', color='r', )
plt.xlabel('Variable Importance')
plt.gca().legend_ = None
plt.savefig(PATH + 'feat_importance.png', dpi=300)
plt.close()

# Decision tree classification, on training set only.
# Use max_depth to limit tree size, since manual pruning is not implemented in scikit-learn.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=12)
tree = DecisionTreeClassifier(max_depth=6)
tree.fit(X_train, y_train)
score = tree.score(X_train, y_train)
print("Training error rate = ", 1-score)
print("Tree node size =", tree.tree_.node_count)
# Prediction
y_pred = tree.predict(X_test)
print('Classification Report:\n', classification_report(y_test, y_pred))
cm = pd.DataFrame(confusion_matrix(y_test, y_pred).T, index=['No', 'Yes'], columns=['No', 'Yes'])
print('Confusion matrix:\n', cm)

# Lab 8.3.2 Fitting Regression Trees
# mass dataset is in R ISLR package
boston = pd.read_csv('boston.csv', index_col=0)
print(boston.head(5))

# Tree regression and prediction
X = boston.drop('medv', axis=1)
y = boston['medv']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=12)
tree_regr = DecisionTreeRegressor(max_depth=3)  # No pruning in scikit-learn
tree_regr.fit(X_train, y_train)

# Visualization
dot_data = export_graphviz(tree_regr, out_file=None, feature_names=X.columns, filled=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png("figs/boston.png")

# Prediction
y_pred = tree_regr.predict(X_test)
print("MSE = ", mean_squared_error(y_test, y_pred))
plt.scatter(y_pred, y_test, label='medv')
plt.plot([0, 1], [0, 1], '--k', transform=plt.gca().transAxes)
plt.xlabel('y_pred')
plt.ylabel('y_test')
plt.savefig(PATH + 'pred.png', dpi=300)
plt.close()

# Feature Importance
importance = pd.DataFrame({'Importance':tree_regr.feature_importances_*100}, index=X.columns)
importance = importance.sort_values('Importance', axis=0, ascending=True)
importance.plot(kind='barh', color='r')
plt.xlabel('Variable Importance')
plt.gca().legend_ = None
plt.savefig(PATH + 'feature_importances_boston.png', dpi=300)
plt.close()

# Lab 8.3.3 Bagging and Random Forests
# Random forest with 13 features
rf13 = RandomForestRegressor(max_features=12, random_state=12)
rf13.fit(X_train, y_train)
y_pred = rf13.predict(X_test)
print("MSE = ", mean_squared_error(y_test, y_pred))
plt.scatter(y_pred, y_test, label='medv')
plt.plot([0, 1], [0, 1], '--k', transform=plt.gca().transAxes)
plt.xlabel('y_pred')
plt.ylabel('y_test')
plt.savefig(PATH + 'random_forest.png', dpi=300)
plt.close()

# Random forest with 6 features
rf6 = RandomForestRegressor(max_features=6, random_state=12)
rf6.fit(X_train, y_train)
y_pred = rf6.predict(X_test)
print("MSE = ", mean_squared_error(y_test, y_pred))
# Feature importance
importance = pd.DataFrame({'Importance':rf6.feature_importances_*100}, index=X.columns)
importance = importance.sort_values('Importance', axis=0, ascending=True)
importance.plot(kind='barh', color='r')
plt.xlabel('Variable Importance')
plt.gca().legend_ = None
plt.savefig(PATH + 'forest_feat_importance.png', dpi=300)
plt.close()

# Lab 8.3.4 Boosting
gbr = GradientBoostingRegressor(n_estimators=500, learning_rate=0.01, random_state=12)
gbr.fit(X_train, y_train)
y_pred = gbr.predict(X_test)
print("MSE = ", mean_squared_error(y_test, y_pred))
# Feature importance
importance = pd.DataFrame({'Importance':gbr.feature_importances_*100}, index=X.columns)
importance = importance.sort_values('Importance', axis=0, ascending=True)
importance.plot(kind='barh', color='r')
plt.xlabel('Variable Importance')
plt.gca().legend_ = None
plt.savefig(PATH + 'boosting.png', dpi=300)
plt.close()
print(importance)
