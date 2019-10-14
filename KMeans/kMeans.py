import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

#eg 1-------------------------------------------

# points
X = np.array([[1,2],[1,4],[1,0],[4,2],[4,4],[4,0]])

#number of clusters you want to create (ie 2 centroid points)
#fit for the X array (passing in data without labels)
kmeans = KMeans(n_clusters=2).fit(X)

#labels show which cluster it belongs to
print(kmeans.labels_)

#pass in array of points and the algoritm predicts which clusters these points these would belong to
print(kmeans.predict([[0,0],[4,4]]))

#x and y coordinate to where the clusters are
print(kmeans.cluster_centers_)











#eg 2-------------------------------------------

#creates random data (200 samples/points and 5 centroids/clusters)
X, y = make_blobs(n_samples=200, centers=5)
print(X.shape, y.shape)

#plotting the data

#starts new graph
plt.figure(0)

plt.grid(True)
#takes everything from first and second column
plt.scatter(X[:,0], X[:,1])
plt.show()

#using Kmeans with our data
clf = KMeans(n_clusters=5)
clf.fit(X)
print(clf.labels_)
z = clf.cluster_centers_
print(z)

plt.scatter(X[:,0], X[:,1], c=clf.labels_)
plt.scatter(z[:,0], z[:,1], c = 'blue')
plt.show()