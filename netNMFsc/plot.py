# plotting functions for netNMF-sc

from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
'''
Select k clusters (silhouette score) and plot silhouette scores
@input 
X: genes x cells numpy array
fname: filename to save plot (optional)
'''
def select_clusters(X,max_clusters=20,fname=''):
	X = X.T
	cluster_range = range( 2, max_clusters )
	avgs = []
	clusters = []
	for n_clusters in cluster_range:
		clusterer = KMeans(n_clusters=n_clusters, random_state=10)
		cluster_labels = clusterer.fit_predict( X )
		clusters.append(cluster_labels)
		silhouette_avg = silhouette_score(X, cluster_labels)
		avgs.append(silhouette_avg)
	k = avgs.index(max(avgs))+2
	print(k,'clusters with average silhouette score:',avgs[k-2])
	return k,clusters[k-2]

'''
perform tSNE plot with clusters
@input 
X: genes x cells numpy array
clusters: list or numpy array, cluster assignments for each cell
fname: filename to save plot (optional)
'''
def tSNE(X,clusters,fname=''):
	X = X.T
	if fname == '':
		fname = 'netNMF-sc_tsne'
	tsne = TSNE().fit_transform(X)
	# choose a color palette with seaborn.
	num_classes = len(np.unique(clusters))
	palette = np.array(sns.color_palette("hls", num_classes))

	# create a scatter plot.
	f = plt.figure(figsize=(8, 8))
	ax = plt.subplot(aspect='equal')
	sc = ax.scatter(tsne[:,0], tsne[:,1], lw=0, s=40, c=palette[clusters.astype(np.int)])
	plt.xlim(-25, 25)
	plt.ylim(-25, 25)
	ax.axis('off')
	ax.axis('tight')
	if fname != '':
		plt.savefig(fname + '.pdf', bbox_inches='tight',format='pdf')
	return ax


'''
clustermap with defined clusters
@input 
X: genes x cells numpy array
clusters: list or numpy array, cluster assignments for each cell
fname: filename to save plot (optional)
'''
def clustermap(X,clusters,fname=''):
	return