# Data Preprocessing
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Mall_Customers_Preprocessed.csv')
df.dropna(inplace=True)
X = df.iloc[:, 3:].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model Selection
from sklearn.cluster import KMeans, Birch, AgglomerativeClustering

kmeans = KMeans(n_clusters=5, random_state=42)
kmedoids = KMedoids(n_clusters=5, random_state=42)
birch = Birch(threshold=0.01, n_clusters=5)

# Model Training
kmeans.fit(X_scaled)
kmedoids.fit(X_scaled)
birch.fit(X_scaled)

# Model Evaluation
from sklearn.metrics import silhouette_score, calinski_harabasz_score

kmeans_silhouette = silhouette_score(X_scaled, kmeans.labels_)
kmedoids_silhouette = silhouette_score(X_scaled, kmedoids.labels_)
birch_silhouette = silhouette_score(X_scaled, birch.labels_)

kmeans_calinski_harabasz = calinski_harabasz_score(X_scaled, kmeans.labels_)
kmedoids_calinski_harabasz = calinski_harabasz_score(X_scaled, kmedoids.labels_)
birch_calinski_harabasz = calinski_harabasz_score(X_scaled, birch.labels_)

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
kmedoids_segment_5 = df[df['kmedoids_cluster'] == 4]

# BIRCH Segments
birch_segment_1 = df[df['birch_cluster'] == 0]
birch_segment_2 = df[df['birch_cluster'] == 1]
birch_segment_3 = df[df['birch_cluster'] == 2]
birch_segment_4 = df[df['birch_cluster'] == 3]
birch_segment_5 = df[df['birch_cluster'] == 4]

# Pass results to Marketing to develop strategies for each segment
print('kmeans_segment_1: ' + kmeans_segment_1)
print('kmeans_segment_2: ' + kmeans_segment_2)
print('kmeans_segment_3: ' + kmeans_segment_3)
print('kmeans_segment_4: ' + kmeans_segment_4)
print('kmeans_segment_5: ' + kmeans_segment_5)
print('kmedoids_segment_1: ' + kmedoids_segment_1)
print('kmedoids_segment_2: ' + kmedoids_segment_2)
print('kmedoids_segment_3: ' + kmedoids_segment_3)
print('kmedoids_segment_4: ' + kmedoids_segment_4)
print('kmedoids_segment_5: ' + kmedoids_segment_5)
print('birch_segment_1: ' + birch_segment_1)
print('birch_segment_2: ' + birch_segment_2)
print('birch_segment_3: ' + birch_segment_3)
print('birch_segment_4: ' + birch_segment_4)
print('birch_segment_5: ' + birch_segment_5)
