import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

filename = 'Mall_Customers.csv'
df = pd.read_csv(filename)

print(df.head())   
print(df.dtypes)

df = df.drop(['CustomerID'],axis = 1).copy()

gender_encoder = LabelEncoder()
gender_encoder.fit(df['Gender'])
df['Gender'] = gender_encoder.transform(df['Gender'])
print(df.head())
print(df['Gender'].value_counts())

distortions = []
for k in range(2,15):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df)
    distortions.append(kmeans.inertia_)
    
fig = plt.figure(figsize=(15,5))
plt.plot(range(2,15), distortions, 'bo-')
plt.grid(True)
plt.title('Elbow curve')
plt.show()


km = KMeans(n_clusters = 3)
clf = km.fit(df)


print(clf.labels_)

z = clf.cluster_centers_

plt.figure(figsize=(30,10))
plt.grid(True)
plt.scatter( x = 'Annual Income (k$)' ,y = 'Spending Score (1-100)' , data = df, c= clf.labels_)
plt.scatter(x = z[:,2], y =z[:,3], c='red')
plt.show()