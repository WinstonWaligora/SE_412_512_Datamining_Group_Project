# Data Preprocessing
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, Birch
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv('Mall_Customers_Preprocessed.csv')
df.dropna(inplace=True)

# Select relevant features and scale data
X = df.iloc[:, [2,3]].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method to find number of clusters
sumof_squared_distance = []
for i in range(1, 12):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    sumof_squared_distance.append(kmeans.inertia_)
plt.plot(range(1,12), sumof_squared_distance)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of squared distance')
plt.savefig('elbow-results.png')
plt.show()
plt.clf()

# Model Selection
kmeans = KMeans(n_clusters=5, random_state=42)
kmedoids = KMedoids(n_clusters=5, random_state=42)
birch = Birch(threshold=0.01, n_clusters=5)

# Model training and prediction
y_kmeans = kmeans.fit_predict(X_scaled)
y_kmedoids = kmedoids.fit_predict(X_scaled)
y_birch = birch.fit_predict(X_scaled)

# Model Evaluation
kmeans_silhouette = silhouette_score(X_scaled, kmeans.labels_)
kmedoids_silhouette = silhouette_score(X_scaled, kmedoids.labels_)
birch_silhouette = silhouette_score(X_scaled, birch.labels_)

print("silhouette score for KMeans:")
print(kmeans_silhouette)
print("silhouette score for KMedoids:")
print(kmedoids_silhouette)
print("silhouette score for Birch:")
print(birch_silhouette)

kmeans_calinski_harabasz = calinski_harabasz_score(X_scaled, kmeans.labels_)
kmedoids_calinski_harabasz = calinski_harabasz_score(X_scaled, kmedoids.labels_)
birch_calinski_harabasz = calinski_harabasz_score(X_scaled, birch.labels_)

print("calinski harabasz for KMeans:")
print(kmeans_calinski_harabasz)
print("calinski harabasz for KMedoids:")
print(kmedoids_calinski_harabasz)
print("calinski harabasz for Birch:")
print(birch_calinski_harabasz)

# Customer Segmentation
df['kmeans_cluster'] = kmeans.labels_
df['kmedoids_cluster'] = kmedoids.labels_
df['birch_cluster'] = birch.labels_

# Visualize clusters
# KMeans
plt.scatter(X_scaled[y_kmeans == 0,0], X_scaled[y_kmeans == 0,1], c='brown', label='Cluster 1')
plt.scatter(X_scaled[y_kmeans == 1,0], X_scaled[y_kmeans == 1,1], c='blue', label='Cluster 2')
plt.scatter(X_scaled[y_kmeans == 2,0], X_scaled[y_kmeans == 2,1], c='green', label='Cluster 3')
plt.scatter(X_scaled[y_kmeans == 3,0], X_scaled[y_kmeans == 3,1], c='cyan', label='Cluster 4')
plt.scatter(X_scaled[y_kmeans == 4,0], X_scaled[y_kmeans == 4,1], c='magenta', label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='red')
plt.title('KMeans Clustering')
plt.xlabel('Spending Score')
plt.ylabel('Income')
plt.plot
plt.savefig('kmeansplot.png')
plt.show()

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
kmedoids_segment_5 = df[df['kmedoids_cluster'] == 4]

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
print('kmedoids_segment_5:')
print(kmedoids_segment_5)
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
