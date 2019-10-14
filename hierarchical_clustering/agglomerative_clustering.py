from sklearn.cluster import AgglomerativeClustering
import math
import numpy as np
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

points = np.array([[3,7], [2,6], [5,8], [6,6], [5,5], [12,8], [10,6], [8,4], [7,3], [2,2], [5,2]])

#This is the condensed matrix of distance between the points:
dc = pdist(points)
print('This is the distance condensed matrix:')
print(dc)

x = np.array([3,2,5,6,5,12,10,8,7,2,5])
y = np.array([7,6,8,6,5,8,6,4,3,2,2])

plt.scatter(x, y)
fig = plt.gcf()
fig.canvas.set_window_title('Scatter Plot of points')
plt.show()

#Fitting the data in Agg Clustering, can use this in future.
clustering = AgglomerativeClustering().fit(points)

AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto',
                        connectivity=None, distance_threshold=None,
                        linkage='single', memory=None, n_clusters=1,
                        pooling_func='deprecated')

S= hierarchy.linkage(dc, 'single')
sdn = hierarchy.dendrogram(S)
fig = plt.gcf()
fig.canvas.set_window_title('Single-Linkage Clustering')
plt.show()

A = hierarchy.linkage(dc, 'average')
adn = hierarchy.dendrogram(A)
fig = plt.gcf()
fig.canvas.set_window_title('Average Linkage Clustering')
plt.show()

C = hierarchy.linkage(dc, 'complete')
cdn = hierarchy.dendrogram(C)
fig = plt.gcf()
fig.canvas.set_window_title('Complete Linkage Clustering')
plt.show()

hierarchy.set_link_color_palette(None)  # reset to default after use
