import  numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import pandas as pd

X,_ = make_blobs(n_samples=300,centers=4,cluster_std=0.90,random_state=0)

f1 = plt.figure(1)
plt.scatter(X[:,0],X[:,1],s=50)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Synthtic data')

kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

distances = np.zeros((X.shape[0],centroids.shape[0]))
for i , cetroid in enumerate(centroids):
    distances[:,i] = np.linalg.norm(X-cetroid,axis=1)

df_distances = pd.DataFrame(distances,columns=[f"Centroid{i + 1}"for i in range(centroids.shape[0])])
df_distances['Assigned Centroid'] = labels + 1
df_distances.index.name = 'Data Point'

print("Table showing distance of each datapoint to each centroid and assigned centroid: ")
print(df_distances)

f2 = plt.figure(2)
plt.scatter(X[:,0],X[:,1],c=labels,s=50,cmap='viridis')
plt.scatter(centroids[:,0], centroids[:, 1],marker='*',s=200, c='red', label='Centroids')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-means Clustering")
plt.legend()

plt.show()