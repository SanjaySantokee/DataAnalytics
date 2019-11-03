# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 23:21:09 2019

@author: shell
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN

X,y = make_blobs(n_samples=200, centers=5)
print(X.shape, y.shape)

plt.figure(0)
plt.grid(True)
plt.scatter(X[:,0], X[:,1])
plt.show()

clf = DBSCAN(eps=3, min_samples=5)
clf.fit(X)
print(clf.labels_)

plt.scatter(X[:,0],X[:,1],c=clf.labels_)
plt.show()


