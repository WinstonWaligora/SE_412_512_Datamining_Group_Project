import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, Birch
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from warnings import simplefilter
from scipy.cluster.hierarchy import dendrogram, linkage

# Ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# Load and preprocess the dataset
df = pd.read_csv('Mall_Customers_Preprocessed.csv')
df.dropna(inplace=True)

# Select relevant features and standardize them
X = df.iloc[:, [2,3]].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method to find the optimal number of clusters for k-means
wcss_kmeans = []
for i in range(1, 12):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss_kmeans.append(kmeans.inertia_)
plt.plot(range(1,12), wcss_kmeans)
plt.title('Elbow Method for Optimal k (KMeans)')
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of squared distance')
plt.savefig('elbow-results-kmeans.png')
plt.show()
plt.clf()

# Elbow method to find the optimal number of clusters for k-medoids
wcss_kmedoids = []
for k in range(1, 12):
    kmedoids = KMedoids(n_clusters=k, random_state=42)
    kmedoids.fit(X_scaled)
    wcss_kmedoids.append(kmedoids.inertia_)
plt.plot(range(1, 12), wcss_kmedoids, marker='o')
plt.title('Elbow Method for Optimal k (KMedoids)')
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of squared distance')
plt.savefig('elbow-results-kmedoids.png')
plt.show()
plt.clf()

# Silhouette score to find the optimal number of clusters for k-medoids
# Look for the value of k that maximizes the silhouette score.
silhouette_scores = []
for k in range(2,12):
    kmedoids = KMedoids(n_clusters=k, random_state=42)
    kmedoids.fit(X_scaled)
    labels = kmedoids.labels_
    silhouette_scores.append(silhouette_score(X_scaled, labels))

plt.plot(range(2,12), silhouette_scores, marker='o')
plt.title('Silhouette Score for K-Medoids Clustering')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.savefig('silhouette-scores-kmedoids')
plt.show()
plt.clf()

# Apply k-means, k-medoids, and birch algorithm to perform clustering and return cluster labels
kmeans = KMeans(n_clusters=5, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)
kmedoids = KMedoids(n_clusters=4, random_state=42)
y_kmedoids = kmedoids.fit_predict(X_scaled)
birch = Birch(threshold=0.5, n_clusters=None)
birch.fit(X_scaled)

# Model Evaluation using Silhouette Score
# Measures how similar an object is to its own cluster compared to other clusters
kmeans_silhouette = silhouette_score(X_scaled, kmeans.labels_)
kmedoids_silhouette = silhouette_score(X_scaled, kmedoids.labels_)
birch_silhouette = silhouette_score(X_scaled, birch.labels_)

print("Silhouette Score for KMeans:")
print(kmeans_silhouette)
print("Silhouette Score for KMedoids:")
print(kmedoids_silhouette)
print("Silhouette Score for Birch:")
print(birch_silhouette)

# Model Evaluation using Calinski-Harabasz Index
# The index aims to capture the compactness of clusters and the separation between them
# Higher values indicate better-defined and well-separated clusters.
# Lower values may suggest that clusters are not well-separated or are too dispersed.
kmeans_calinski_harabasz = calinski_harabasz_score(X_scaled, kmeans.labels_)
kmedoids_calinski_harabasz = calinski_harabasz_score(X_scaled, kmedoids.labels_)
birch_calinski_harabasz = calinski_harabasz_score(X_scaled, birch.labels_)

print("Calinski-Harabasz Index for KMeans:")
print(kmeans_calinski_harabasz)
print("Calinski-Harabasz Index for KMedoids:")
print(kmedoids_calinski_harabasz)
print("Calinski-Harabasz Index for Birch:")
print(birch_calinski_harabasz)

# Model evaluation using Davies-Bouldin Index
# It measures the compactness and separation of clusters in a partitioned dataset
# Lower values are preferable, indicating more cohesive and well-separated clusters.
# Higher values suggest that clusters might be less compact or less well-separated
kmeans_davies_bouldin_score = davies_bouldin_score(X_scaled, kmeans.labels_)
kmedoids_davies_bouldin_score = davies_bouldin_score(X_scaled, kmedoids.labels_)
birch_davies_bouldin_score = davies_bouldin_score(X_scaled, birch.labels_)

print("Davies-Bouldin Index for KMeans:")
print(kmeans_davies_bouldin_score)
print("Davies-Bouldin Index for KMedoids:")
print(kmedoids_davies_bouldin_score)
print("Davies-Bouldin Index for Birch:")
print(birch_davies_bouldin_score)

# Visualize clusters
# KMeans
plt.scatter(X_scaled[y_kmeans == 0,0], X_scaled[y_kmeans == 0,1], c='brown', label='Cluster 1')
plt.scatter(X_scaled[y_kmeans == 1,0], X_scaled[y_kmeans == 1,1], c='blue', label='Cluster 2')
plt.scatter(X_scaled[y_kmeans == 2,0], X_scaled[y_kmeans == 2,1], c='green', label='Cluster 3')
plt.scatter(X_scaled[y_kmeans == 3,0], X_scaled[y_kmeans == 3,1], c='cyan', label='Cluster 4')
plt.scatter(X_scaled[y_kmeans == 4,0], X_scaled[y_kmeans == 4,1], c='magenta', label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='red')
plt.title('KMeans Clustering')
plt.xlabel('Income')
plt.ylabel('Spending Score')
plt.plot
plt.savefig('kmeansplot.png')
plt.show()
plt.clf()

# KMedoids
plt.scatter(X_scaled[y_kmedoids == 0,0], X_scaled[y_kmedoids == 0,1], c='brown', label='Cluster 1')
plt.scatter(X_scaled[y_kmedoids == 1,0], X_scaled[y_kmedoids == 1,1], c='blue', label='Cluster 2')
plt.scatter(X_scaled[y_kmedoids == 2,0], X_scaled[y_kmedoids == 2,1], c='green', label='Cluster 3')
plt.scatter(X_scaled[y_kmedoids == 3,0], X_scaled[y_kmedoids == 3,1], c='cyan', label='Cluster 4')
plt.scatter(X_scaled[y_kmedoids == 4,0], X_scaled[y_kmedoids == 4,1], c='magenta', label='Cluster 5')
plt.scatter(kmedoids.cluster_centers_[:,0], kmedoids.cluster_centers_[:,1], c='red')
plt.title('KMedoids Clustering')
plt.xlabel('Income')
plt.ylabel('Spending Score')
plt.plot
plt.savefig('kmedoidsplot.png')
plt.show()
plt.clf()

# Birch
# Create linkage matrix for hierarchical clustering
linkage_matrix = linkage(X_scaled, method='ward')

# Plot dendrogram for BIRCH algorithm
plt.figure(figsize=(10, 5))
dendrogram(linkage_matrix, labels=birch.labels_)
plt.title('Dendrogram for Birch Clustering')
# X-axis represents the individual data points
plt.xlabel('Data Points')
# Height represents the dissimilarity or distance between the clusters.
plt.ylabel('Distance')
plt.savefig('birchdendrogram.png')
plt.show()
plt.clf()

# Customer Segmentation
df['kmeans_cluster'] = kmeans.labels_
df['kmedoids_cluster'] = kmedoids.labels_
df['birch_cluster'] = birch.labels_

# Marketing Strategy
# KMeans Segments
kmeans_segment_1 = df[df['kmeans_cluster'] == 0]
kmeans_segment_2 = df[df['kmeans_cluster'] == 1]
kmeans_segment_3 = df[df['kmeans_cluster'] == 2]
kmeans_segment_4 = df[df['kmeans_cluster'] == 3]
kmeans_segment_5 = df[df['kmeans_cluster'] == 4]

# KMedoids Segments
kmedoids_segment_1 = df[df['kmedoids_cluster'] == 0]
kmedoids_segment_2 = df[df['kmedoids_cluster'] == 1]
kmedoids_segment_3 = df[df['kmedoids_cluster'] == 2]
kmedoids_segment_4 = df[df['kmedoids_cluster'] == 3]

# BIRCH Segments
birch_segment_1 = df[df['birch_cluster'] == 0]
birch_segment_2 = df[df['birch_cluster'] == 1]
birch_segment_3 = df[df['birch_cluster'] == 2]
birch_segment_4 = df[df['birch_cluster'] == 3]
birch_segment_5 = df[df['birch_cluster'] == 4]

# Pass results to Marketing to develop strategies for each segment
print('kmeans_segment_1:')
print(kmeans_segment_1)
print('kmeans_segment_2:')
print(kmeans_segment_2)
print('kmeans_segment_3:')
print(kmeans_segment_3)
print('kmeans_segment_4:')
print(kmeans_segment_4)
print('kmeans_segment_5:')
print(kmeans_segment_5)
print('kmedoids_segment_1:')
print(kmedoids_segment_1)
print('kmedoids_segment_2:')
print(kmedoids_segment_2)
print('kmedoids_segment_3:')
print(kmedoids_segment_3)
print('kmedoids_segment_4:')
print(kmedoids_segment_4)
print('birch_segment_1:')
print(birch_segment_1)
print('birch_segment_2:')
print(birch_segment_2)
print('birch_segment_3:')
print(birch_segment_3)
print('birch_segment_4:')
print(birch_segment_4)
print('birch_segment_5:')
print(birch_segment_5)
