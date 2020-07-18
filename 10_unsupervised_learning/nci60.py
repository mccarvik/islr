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

# NCI60 dataset is in R ISLR package
nci60_data  = pd.read_csv('NCI60.csv')
nci60_data.drop("Unnamed: 0", axis=1, inplace=True)
nci60_data.drop("labs", axis=1, inplace=True)
nci60_labs = pd.read_csv('NCI60_labels.csv', header=None)

print(nci60_data.head(5))
print(nci60_labs.head(5))
nci60_labs[0] = nci60_labs[0].apply(lambda x: x.strip())
nci60_data.info()
print(nci60_labs[0].value_counts())

# PCA
pca = PCA()
nci60_pca = pd.DataFrame(pca.fit_transform(nci60_data))
# Plot
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,6))
color_idx = pd.factorize(nci60_labs[0])[0]
cmap = plt.cm.hsv
# Left plot
ax1.scatter(nci60_pca.iloc[:,0], nci60_pca.iloc[:,1], c=color_idx, cmap=cmap, alpha=0.5, s=50)
ax1.set_ylabel('Principal Component 2')
# Right plot
ax2.scatter(nci60_pca.iloc[:,0], nci60_pca.iloc[:,2], c=color_idx, cmap=cmap, alpha=0.5, s=50)
ax2.set_ylabel('Principal Component 3')
# Custom legend for the classes since we do not create scatter plots per class (which could have their own labels).
handles = []
labels = pd.factorize(nci60_labs[0].unique())
norm = mpl.colors.Normalize(vmin=0.0, vmax=14.0)

for i, v in zip(labels[0], labels[1]):
    handles.append(mpl.patches.Patch(color=cmap(norm(i)), label=v, alpha=0.5))
ax2.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

# xlabel for both plots
for ax in fig.axes:
    ax.set_xlabel('Principal Component 1')
plt.savefig(PATH + 'pca_nci.png', dpi=300)
plt.close()

pd.DataFrame([nci60_pca.iloc[:,:5].std(axis=0, ddof=0).as_matrix(),
              pca.explained_variance_ratio_[:5],
              np.cumsum(pca.explained_variance_ratio_[:5])],
             index=['Standard Deviation', 'Proportion of Variance', 'Cumulative Proportion'],
             columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])

nci60_pca.iloc[:,:10].var(axis=0, ddof=0).plot(kind='bar', rot=0)
plt.ylabel('Variances');
plt.savefig(PATH + 'nci_var.png', dpi=300)
plt.close()

# scree plot
fig , (ax1,ax2) = plt.subplots(1,2, figsize=(15,5))

# Left plot
ax1.plot(pca.explained_variance_ratio_, '-o')
ax1.set_ylabel('Proportion of Variance Explained')
ax1.set_ylim(ymin=-0.01)

# Right plot
ax2.plot(np.cumsum(pca.explained_variance_ratio_), '-ro')
ax2.set_ylabel('Cumulative Proportion of Variance Explained')
ax2.set_ylim(ymax=1.05)

for ax in fig.axes:
    ax.set_xlabel('Principal Component')
    ax.set_xlim(-1,65)
plt.savefig(PATH + 'nci_proportion_explained.png', dpi=300)
plt.close()

# Clustering
fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(20,20))
for linkage, cluster, ax in zip([hierarchy.complete(nci60_data), hierarchy.average(nci60_data), hierarchy.single(nci60_data)],
                                ['c1','c2','c3'],
                                [ax1,ax2,ax3]):
    cluster = hierarchy.dendrogram(linkage, labels=nci60_data.index, orientation='right', color_threshold=0, leaf_font_size=10, ax=ax)

ax1.set_title('Complete Linkage')
ax2.set_title('Average Linkage')
ax3.set_title('Single Linkage');
plt.savefig(PATH + 'nci_clusters.png', dpi=300)
plt.close()

# Cut dendrogram with complete linkage
plt.figure(figsize=(10,20))
cut4 = hierarchy.dendrogram(hierarchy.complete(nci60_data),
                            labels=nci60_data.index, orientation='right', color_threshold=140, leaf_font_size=10)
plt.vlines(140,0,plt.gca().yaxis.get_data_interval()[1], colors='r', linestyles='dashed')
plt.savefig(PATH + 'nci_clusters_with_cut.png', dpi=300)
plt.close()

# KMeans
np.random.seed(2)
km_nci60 = KMeans(n_clusters=4, n_init=50)
km_nci60.fit(nci60_data)
print(km_nci60.labels_)
# Observations per KMeans cluster
print(pd.Series(km_nci60.labels_).value_counts().sort_index())
# Observations per Hierarchical cluster
nci60_cut = hierarchy.dendrogram(hierarchy.complete(nci60_data), truncate_mode='lastp', p=4, show_leaf_counts=True)

# Hierarchy based on Principal Components 1 to 5
plt.figure(figsize=(10,20))
pca_cluster = hierarchy.dendrogram(hierarchy.complete(nci60_pca.iloc[:,:5]), labels=nci60_labs[0].values, orientation='right', color_threshold=100, leaf_font_size=10)
plt.savefig(PATH + 'highlighted_clusters.png', dpi=300)
plt.close()
hierarchy.dendrogram(hierarchy.complete(nci60_pca), truncate_mode='lastp', p=4, show_leaf_counts=True)
plt.savefig(PATH + 'short_dendogram.png', dpi=300)
plt.close()