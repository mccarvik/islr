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

usarrests = pd.read_csv('usarrests.csv')
print(usarrests.info())
usarrests = usarrests.set_index("Unnamed: 0")
# usarrests.drop("Unnamed: 0", axis=1, inplace=True)
# Mean and variance
print(usarrests.mean())
print(usarrests.var())

# Standardize the data
X = pd.DataFrame(scale(usarrests), index=usarrests.index, columns=usarrests.columns)
print(X.mean())
print(X.var())

# Principal Component Analysis
pca = PCA()
usarrests_loadings = pd.DataFrame(pca.fit(X).components_.T, index=usarrests.columns, columns=['PC1', 'PC2', 'PC3', 'PC4'])
print(usarrests_loadings)

usarrests_score = pd.DataFrame(pca.fit_transform(X), index=X.index, columns=['PC1', 'PC2', 'PC3', 'PC4'])
print(usarrests_score)

# Standard deviation, Variance, and EVR of principal components
usarrests_score_stdvar = pd.DataFrame([np.sqrt(pca.explained_variance_), pca.explained_variance_, pca.explained_variance_ratio_], index=['STDEV', 'VAR', 'Explained VAR Ratio'], columns=['PC1', 'PC2', 'PC3', 'PC4'])
print(usarrests_score_stdvar)

mpl.style.use('seaborn-white')
fig , ax1 = plt.subplots(figsize=(9,7))

ax1.set_xlim(-3.5,3.5)
ax1.set_ylim(-3.5,3.5)

# Plot Principal Components 1 and 2
for i in usarrests_score.index:
    ax1.annotate(i, (usarrests_score.PC1.loc[i], -usarrests_score.PC2.loc[i]), ha='center')

# Plot reference lines
ax1.hlines(0,-3.5,3.5, linestyles='dotted', colors='grey')
ax1.vlines(0,-3.5,3.5, linestyles='dotted', colors='grey')

ax1.set_xlabel('First Principal Component')
ax1.set_ylabel('Second Principal Component')
    
# Plot Principal Component loading vectors, using a second y-axis.
ax2 = ax1.twinx().twiny() 

ax2.set_ylim(-1,1)
ax2.set_xlim(-1,1)
ax2.tick_params(axis='y', colors='orange')
ax2.set_xlabel('Principal Component loading vectors', color='orange')

# Plot labels for vectors. Variable 'a' is a small offset parameter to separate arrow tip and text.
a = 1.07  
for i in usarrests_loadings[['PC1', 'PC2']].index:
    ax2.annotate(i, (usarrests_loadings.PC1.loc[i]*a, -usarrests_loadings.PC2.loc[i]*a), color='orange')

# Plot vectors
ax2.arrow(0,0,usarrests_loadings.PC1[0], -usarrests_loadings.PC2[0], width=0.006)
ax2.arrow(0,0,usarrests_loadings.PC1[1], -usarrests_loadings.PC2[1], width=0.006)
ax2.arrow(0,0,usarrests_loadings.PC1[2], -usarrests_loadings.PC2[2], width=0.006)
ax2.arrow(0,0,usarrests_loadings.PC1[3], -usarrests_loadings.PC2[3], width=0.006)
plt.savefig(PATH + 'pca1.png', dpi=300)
plt.close()

mpl.style.use('ggplot')
plt.figure(figsize=(7,5))

plt.plot([1,2,3,4], pca.explained_variance_ratio_, '-o', label='Individual component')
plt.plot([1,2,3,4], np.cumsum(pca.explained_variance_ratio_), '-s', label='Cumulative')

plt.ylabel('Proportion of Variance Explained')
plt.xlabel('Principal Component')
plt.xlim(0.75,4.25)
plt.ylim(0,1.05)
plt.xticks([1,2,3,4])
plt.legend(loc=2);
plt.savefig(PATH + 'pca2.png', dpi=300)
plt.close()
