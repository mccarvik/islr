import pdb, sys
# Math and data processing
import numpy as np
import scipy as sp
import pandas as pd

# scikit-learn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

# scipy
from scipy.cluster import hierarchy

# Visulization
from IPython.display import display
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use('ggplot')
PATH = '/home/ec2-user/environment/islr/10_unsupervised_learning/figs/'

# Generate data
np.random.seed(2)
X = np.random.standard_normal((50,2))
X[:25,0] = X[:25,0]+3
X[:25,1] = X[:25,1]-4

# K = 2
km1 = KMeans(n_clusters=2, n_init=20)
km1.fit(X)
print(km1.labels_)

# K = 3
np.random.seed(4)
km2 = KMeans(n_clusters=3, n_init=20)
km2.fit(X)
print(pd.Series(km2.labels_).value_counts())
print(km2.cluster_centers_)
print(km2.labels_)

# Sum of distances of samples to their closest cluster center.
print(km2.inertia_)

# Plots
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,5))

ax1.scatter(X[:,0], X[:,1], s=40, c=km1.labels_, cmap=plt.cm.prism) 
ax1.set_title('K-Means Clustering Results with K=2')
ax1.scatter(km1.cluster_centers_[:,0], km1.cluster_centers_[:,1], marker='+', s=100, c='k', linewidth=2)

ax2.scatter(X[:,0], X[:,1], s=40, c=km2.labels_, cmap=plt.cm.prism) 
ax2.set_title('K-Means Clustering Results with K=3')
ax2.scatter(km2.cluster_centers_[:,0], km2.cluster_centers_[:,1], marker='+', s=100, c='k', linewidth=2);
plt.savefig(PATH + 'kmeans.png', dpi=300)
plt.close()