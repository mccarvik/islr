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

# Generate set that is linearly seperable
np.random.seed(1)
X = np.random.normal(0, 1, (20, 2))
y = np.concatenate((np.repeat(0, 10), np.repeat(1, 10)))
X[y==1, :] = X[y==1, :] + 1.3

df = pd.concat([pd.DataFrame(data=X, columns=['x1', 'x2']), pd.Series(y, name='y')], axis=1)
plt.figure(figsize=(10, 10))
sns.scatterplot(x='x1', y='x2', hue='y', data=df)
plt.savefig(PATH + 'lin_sep_rand.png', dpi=300)
plt.close()
