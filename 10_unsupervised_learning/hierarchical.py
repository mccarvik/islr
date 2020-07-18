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


X = np.random.standard_normal((50,2))
X[:25,0] = X[:25,0]+3
X[:25,1] = X[:25,1]-4
fig, (ax1,ax2,ax3) = plt.subplots(3,1, figsize=(15,18))

for linkage, cluster, ax in zip([hierarchy.complete(X), hierarchy.average(X), hierarchy.single(X)], ['c1','c2','c3'],
                                [ax1,ax2,ax3]):
    cluster = hierarchy.dendrogram(linkage, ax=ax, color_threshold=0)

ax1.set_title('Complete Linkage')
ax2.set_title('Average Linkage')
ax3.set_title('Single Linkage');
plt.savefig(PATH + 'hierarchy.png', dpi=300)
plt.close()