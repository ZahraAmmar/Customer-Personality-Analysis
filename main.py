
# importing libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA


# loading the dataset

df = pd.read_csv("/content/customer personality analysis.csv")


## Data Exploration
"""

df.head()

df.info()

df.columns

df.describe()

# Count of unique values for categorical features
categorical_features = ['Education', 'Marital Status']
categorical_unique_counts = {col: df[col].nunique() for col in categorical_features}

print(categorical_unique_counts)

for col in categorical_features:
  print(df[col].unique())

# Frequency distribution of categorical features
categorical_frequency = {col: df[col].value_counts() for col in categorical_features}

print(categorical_frequency)

df.isnull().sum()

"""## Summary Statistics and Potential Preprocessing Steps

###Numerical Features:  
- Income: Contains 24 missing values. These need to be handled, possibly through imputation.  
- Year_Birth: The minimum year of birth is 1893, which might indicate outliers or erroneous data.  
- *Mnt (Product spending categories)**: These features have a wide range of values and may benefit from normalization or standardization.  
- *Num (Purchase counts)**: Similar to spending categories, these might require normalization, especially if used in distance-based clustering algorithms.


###Categorical Features:  
- Education: Has 5 unique categories. Consider encoding (e.g., one-hot encoding) for clustering algorithms.  
- Marital_Status: Has 8 unique categories, including less common statuses like 'Alone', 'Absurd', and 'YOLO'. These might need to be grouped or encoded differently.  

###Date Feature:  
Dt_Customer: The enrollment date could be transformed into a more useful feature, like tenure or days since enrollment.  


###Binary Features:  
AcceptedCmp1-5, Response, Complain: Already in a binary format, suitable for most clustering algorithms.  

###Constant Features:  
Z_CostContact, Z_Revenue: These appear to be constant and might not be useful for clustering. Consider dropping them unless they have a specific meaning or use.  


###Frequency Distribution of Categorical Features:  
Education and Marital_Status show varied distributions, which could be insightful for clustering. However, small categories in Marital_Status may need special treatment.

###Missing Values:  
Income: Needs imputation or removal of missing values.

## Data Preprocessing

### Dropping missing values and constant columns
"""

# Dropping constant features: Z_CostContact and Z_Revenue
df_dropped = df.drop(columns=['Z_CostContact', 'Z_Revenue'])

# Handling missing values in the 'Income' column by dropping rows with missing income
df_cleaned = df_dropped.dropna(subset=['Income'])

# Displaying the first few rows of the cleaned dataframe
df_cleaned.head()

"""### Identifying and dealing with outliers"""

# Exploring the 'Year_Birth' column for potential outliers
year_birth_stats = df_cleaned['Year Birth'].describe()

# Identifying potential outliers using IQR method
Q1 = year_birth_stats['25%']
Q3 = year_birth_stats['75%']
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Counting outliers
outliers = df_cleaned[(df_cleaned['Year Birth'] < lower_bound) | (df_cleaned['Year Birth'] > upper_bound)]
outlier_count = outliers.shape[0]

print(year_birth_stats)
print(lower_bound)
print(upper_bound)
print(outlier_count)
outliers['Year Birth'].sort_values().unique()

# Printing the rows with the outliers in the 'Year_Birth' column
outlier_rows = df_cleaned[df_cleaned['Year Birth'].isin([1893, 1899, 1900])]
outlier_rows

# Dropping the rows with outliers in the 'Year_Birth' column
df_cleaned_no_outliers = df_cleaned.drop(outlier_rows.index)

# Verifying the removal by checking if these rows still exist in the cleaned dataframe
verification = df_cleaned_no_outliers[df_cleaned_no_outliers['Year Birth'].isin([1893, 1899, 1900])]
verification.empty, df_cleaned_no_outliers.shape

"""### Dealing with issues in categorical columns"""

# Dealing with issues in Marital_Status column

marital_status_counts = df_cleaned_no_outliers['Marital Status'].value_counts()

marital_status_counts

# Replace "Alone" with "Single" in Marital_Status
df_cleaned_no_outliers['Marital Status'].replace('Alone', 'Single', inplace=True)

# Drop rows where Marital_Status is "Absurd" or "YOLO"
df_cleaned_no_outliers = df_cleaned_no_outliers[~df_cleaned_no_outliers['Marital Status'].isin(['Absurd', 'YOLO'])]

# New value counts for Marital_Status after the replacements and drops
new_marital_status_counts = df_cleaned_no_outliers['Marital Status'].value_counts()

new_marital_status_counts

# Previewing value counts in Education
education_counts = df_cleaned_no_outliers['Education'].value_counts()

education_counts

"""### Dealing with Date Feature: Dt_Customer

In the code below the column Dt_Customer has been converted to Datetime format, then earliest and last date is identified     
Then another feature has been created Customer_Tenure_Days which shows number of days since the customer started using the service
"""

# Convert Dt_Customer column to datetime format (assuming the format is day-month-year)
df_cleaned_no_outliers['Dt_Customer'] = pd.to_datetime(df_cleaned_no_outliers['Dt_Customer'], format='%d-%m-%Y', errors='coerce')

# Handling if there are any NaT (Not a Time) values after conversion
df_cleaned_no_outliers.dropna(subset=['Dt_Customer'], inplace=True)

# Finding the earliest and latest date in the Dt_Customer column
earliest_date = df_cleaned_no_outliers['Dt_Customer'].min()
latest_date = df_cleaned_no_outliers['Dt_Customer'].max()

# Calculating customer tenure as days from Dt_Customer to the current date
current_date = datetime.now()
df_cleaned_no_outliers['Customer_Tenure_Days'] = (current_date - df_cleaned_no_outliers['Dt_Customer']).dt.days

# Displaying the earliest and latest date, and the first few rows to see the new column
earliest_date, latest_date, df_cleaned_no_outliers[['Dt_Customer', 'Customer_Tenure_Days']].head()

df_cleaned_no_outliers.insert(5, "No_of_kid", df_cleaned_no_outliers['Kidhome'] + df_cleaned_no_outliers['Teenhome'])
df_cleaned_no_outliers.head()



"""### Label Encoding"""

marital_status_encoded = pd.get_dummies(df_cleaned_no_outliers['Marital Status'], prefix='Marital')
df_encoded = df_cleaned_no_outliers.join(marital_status_encoded)

education_hierarchy = {'Basic': 1, 'Graduation': 2, '2n Cycle': 3, 'Master': 4, 'PhD': 5}
df_encoded['Education_Encoded'] = df_encoded['Education'].map(education_hierarchy)

"""### Dropping columns we don't need anymore"""

# Since we have encoded marital status and education also created a useful feature from Dt Customer, we can drop these
df_final = df_encoded.drop(columns=['Kidhome','Teenhome','Marital Status', 'Education', 'Dt_Customer'])

df_final.head()

"""### Data Transformation"""

# Columns to be visualized
columns_to_visualize = ['Income', 'Recency', 'Customer_Tenure_Days', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']

# Plotting distributions
plt.figure(figsize=(15, 10))
for i, col in enumerate(columns_to_visualize):
    plt.subplot(4, 4, i+1)
    sns.histplot(df_cleaned_no_outliers[col], kde=True)
    plt.title(col)
plt.tight_layout()
plt.show()

# Creating separate dataframes for standardization and normalization
df_standardized = df_final.copy()
df_normalized = df_final.copy()

# List of columns to be scaled
columns_to_scale = ['Income', 'Recency', 'Customer_Tenure_Days', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']

# Applying standardization
scaler = StandardScaler()
df_standardized[columns_to_scale] = scaler.fit_transform(df_standardized[columns_to_scale])

# Applying normalization
normalizer = MinMaxScaler()
df_normalized[columns_to_scale] = normalizer.fit_transform(df_normalized[columns_to_scale])

"""## Clustering

### Centroid-based (K-Means):

K-Means works well with standardized data because it minimizes variance within clusters and is sensitive to the scale of the data.  

using df_standardized
"""

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)
kmeans.fit(df_standardized)
labels_kmeans = kmeans.labels_

pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
reduced_data = pca.fit_transform(df_standardized)

plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels_kmeans)  # Assuming labels_kmeans are your K-Means cluster labels
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=200, c='red')  # Plot centroids
plt.title("K-Means Clustering")
plt.show()

print("Centroids:\n", kmeans.cluster_centers_)

"""Internal evaluation"""

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

silhouette = silhouette_score(df_standardized, labels_kmeans)  # Replace with appropriate labels
calinski_harabasz = calinski_harabasz_score(df_standardized, labels_kmeans)
davies_bouldin = davies_bouldin_score(df_standardized, labels_kmeans)

print("Silhouette Score:", silhouette)
print("Calinski-Harabasz Index:", calinski_harabasz)
print("Davies-Bouldin Index:", davies_bouldin)

"""**Silhouette Score:** 0.598  

The Silhouette Score ranges from -1 to 1. A score closer to 1 indicates that data points are well clustered and far from neighboring clusters. A score of 0.598 suggests that the clusters are reasonably well defined and separated from each other.

**Calinski-Harabasz Index:** 9028.17  
The Calinski-Harabasz Index is a measure of cluster validity. A higher score usually indicates that the clusters are dense and well separated, which is desirable in clustering. Your score is quite high, suggesting good clustering performance.

**Davies-Bouldin Index:** 0.495  
This index indicates the average 'similarity' between clusters, where lower values mean better separation between clusters. A value of 0.495, being on the lower side, indicates good clustering where each cluster is distinct from others.

**Interpretation**  
These metrics collectively suggest that your K-Means clustering has performed well in segmenting the data into distinct, well-separated groups.
The relatively high Calinski-Harabasz Index and low Davies-Bouldin Index, along with a decent Silhouette Score, indicate that the clusters are compact and well separated from each other.

#### Finding the optimal value of k

**Elbow Method**

- This graph plots the inertia (within-cluster sum of squares) against the number of clusters. Inertia decreases as the number of clusters increases because the data points are closer to the centroids they are assigned to.  
- The "elbow" point in the graph is where the rate of decrease in inertia sharply changes. This point is typically considered a good trade-off between the number of clusters and the sum of the distances of points from their nearest cluster center.  
- In your elbow method graph, the inertia decreases rapidly until around 3 or 4 clusters and then levels off. This suggests that the optimal number of clusters could be around 3 or 4, as additional clusters beyond this point do not significantly contribute to a decrease in inertia.
"""

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertia = []
for n in range(1, 11):  # Example: testing 1 to 10 clusters
    kmeans = KMeans(n_clusters=n, random_state=42).fit(df_standardized)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

"""**Silhouette Method**

- This graph shows the silhouette score for different numbers of clusters. The silhouette score measures how similar a data point is to its own cluster (cohesion) compared to other clusters (separation).  
- A higher silhouette score indicates better defined clusters. You generally look for the number of clusters that gives the highest silhouette score.  
- In your silhouette method graph, the highest silhouette score occurs at 2 clusters. The score decreases as the number of clusters increases, which implies that 2 clusters might be the optimal number based on the silhouette score.  
"""

from sklearn.metrics import silhouette_score

silhouette_scores = []
for n in range(2, 11):  # Silhouette score is not defined for n=1
    kmeans = KMeans(n_clusters=n, random_state=42).fit(df_standardized)
    score = silhouette_score(df_standardized, kmeans.labels_)
    silhouette_scores.append(score)

plt.plot(range(2, 11), silhouette_scores)
plt.title('Silhouette Method')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()

"""Now we will test if applying PCA before applying clustering can create better clusters and result in better evaluation scores

### Principal Component Analysis (PCA)

Dimensionality reduction can be a crucial step before applying clustering models, especially when dealing with high-dimensional data. Here are some considerations to help you decide whether to apply dimensionality reduction:

**Curse of Dimensionality:**  
High-dimensional data can suffer from the "curse of dimensionality," where the volume of the space increases so much that the available data becomes sparse. This sparsity is problematic for any method that requires statistical significance.

**Noise Reduction:**  
Dimensionality reduction can help to remove noise from the data by retaining only the most significant features, which can improve the performance of clustering algorithms.

**Computational Efficiency:**  
Algorithms like K-Means become computationally expensive as the number of dimensions grows. Reducing the dimensionality can make these algorithms run faster.

**Visualization:**  
It's often useful to visualize the results of clustering to understand the distribution and relationships in your data. Dimensionality reduction is a must for visualizing high-dimensional data in two or three dimensions.

**Interpretability:**  
Fewer dimensions can make the model more interpretable since each dimension can be examined and understood by humans.

It's important to choose the number of components wisely. A good way to decide n_components is to look at the explained variance ratio from PCA and select a number of components that capture a high percentage of the variance in the data. You can plot the cumulative explained variance against the number of components to find an 'elbow' where additional components do not add much explanatory power.

We will explore what would be the ideal number of components to choose from
"""

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Assuming df_standardized is your standardized dataframe
pca = PCA().fit(df_standardized)

# Plotting the cumulative explained variance against the number of components
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
plt.title('Explained Variance by Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()

"""Now we will apply PCA"""

# Perform PCA on standardized data
n_components_pca = 3  # This is based on the elbow point from the PCA variance graph
pca = PCA(n_components=n_components_pca)
df_standardized_pca = pca.fit_transform(df_standardized)

"""Now we can apply k means clustering the same way we did previously and do a comparison

### K means Clustering after PCA
"""

# Perform KMeans clustering on the PCA-reduced data
kmeans_pca = KMeans(n_clusters=3)
kmeans_pca.fit(df_standardized_pca)
labels_kmeans_pca = kmeans_pca.labels_

# Visualize the 3D scatter plot for clusters obtained from PCA-reduced data
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_standardized_pca[:, 0], df_standardized_pca[:, 1], df_standardized_pca[:, 2], c=labels_kmeans_pca, cmap='viridis', marker='o')
centroids_pca = kmeans_pca.cluster_centers_
ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], centroids_pca[:, 2], s=200, c='red', marker='X')
ax.set_title('3D K-Means Clustering (PCA-reduced data)')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
plt.show()

# Calculate and print evaluation metrics for clusters obtained from PCA-reduced data
silhouette_pca = silhouette_score(df_standardized_pca, labels_kmeans_pca)
calinski_harabasz_pca = calinski_harabasz_score(df_standardized_pca, labels_kmeans_pca)
davies_bouldin_pca = davies_bouldin_score(df_standardized_pca, labels_kmeans_pca)
print(f'Silhouette Score (PCA): {silhouette_pca:.3f}')
print(f'Calinski-Harabasz Index (PCA): {calinski_harabasz_pca:.3f}')
print(f'Davies-Bouldin Index (PCA): {davies_bouldin_pca:.3f}')

"""Similar scores mean there was lots of noise in the data, so its better to apply PCA in such cases.

Given that PCA can help with visualization and may improve computational efficiency, you could opt to use the PCA-reduced data for ease of interpretation and further analysis, especially if you need to visualize the clusters or work with a very high-dimensional dataset.
"""

# If you want to visualize the silhouette scores for different numbers of clusters
inertia_pca = []
silhouette_scores_pca = []
for n in range(2, 11):
    kmeans_temp = KMeans(n_clusters=n, random_state=42).fit(df_standardized_pca)
    inertia_pca.append(kmeans_temp.inertia_)
    silhouette_scores_pca.append(silhouette_score(df_standardized_pca, kmeans_temp.labels_))

# Plotting the elbow method graph for PCA-reduced data
plt.plot(range(2, 11), inertia_pca)
plt.title('Elbow Method with PCA')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


# Plotting the silhouette scores graph for PCA-reduced data
plt.plot(range(2, 11), silhouette_scores_pca)
plt.title('Silhouette Method with PCA')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()

"""How to view the clusters?"""

# Assuming df_cleaned_no_outliers is your original dataset before encoding, standardization, and PCA
df_with_clusters_no_pca = df_cleaned_no_outliers.copy()
df_with_clusters_no_pca['Cluster_Labels_No_PCA'] = labels_kmeans  # Labels from KMeans without PCA

df_with_clusters_pca = df_cleaned_no_outliers.copy()
df_with_clusters_pca['Cluster_Labels_PCA'] = labels_kmeans_pca  # Labels from KMeans with PCA

# Analyzing clusters without PCA
cluster_summary_no_pca = df_with_clusters_no_pca.groupby('Cluster_Labels_No_PCA').mean()
print("Cluster Summary without PCA:\n", cluster_summary_no_pca)

# Analyzing clusters with PCA
cluster_summary_pca = df_with_clusters_pca.groupby('Cluster_Labels_PCA').mean()
print("Cluster Summary with PCA:\n", cluster_summary_pca)

df_with_clusters_no_pca.head()

df_with_clusters_pca.head()

"""## The following are some more clustering methods

### Connectivity-based (Hierarchical Clustering):

Hierarchical Clustering can also benefit from standardized data, particularly if the features have different scales.

using df_standardized
"""

# without pca

from scipy.cluster.hierarchy import dendrogram, linkage
linked = linkage(df_standardized, method='ward')
dendrogram(linked)
plt.show()

# with PCA

from scipy.cluster.hierarchy import dendrogram, linkage

# Perform hierarchical clustering on PCA-reduced data
linked_pca = linkage(df_standardized_pca, method='ward')

# Plotting the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linked_pca)
plt.title('Hierarchical Clustering Dendrogram (PCA)')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

from scipy.cluster.hierarchy import fcluster

# Determine the clusters at the specified distance threshold
distance_threshold = 50000  # Replace with the appropriate distance based on your dendrogram
cluster_labels = fcluster(linked_pca, distance_threshold, criterion='distance')

# Assuming df_standardized_pca is the PCA-reduced data used for clustering
silhouette_avg = silhouette_score(df_standardized_pca, cluster_labels)
calinski_harabasz = calinski_harabasz_score(df_standardized_pca, cluster_labels)
davies_bouldin = davies_bouldin_score(df_standardized_pca, cluster_labels)

print("Silhouette Score:", silhouette_avg)
print("Calinski-Harabasz Index:", calinski_harabasz)
print("Davies-Bouldin Index:", davies_bouldin)

# Assuming df_cleaned_no_outliers is your original dataset
df_cleaned_no_outliers['Cluster'] = cluster_labels

cluster_groups = df_cleaned_no_outliers.groupby('Cluster')

# Get means or other statistics for each cluster
cluster_means = cluster_groups.mean()
print(cluster_means)

# You can also examine other statistics like median, count, or standard deviation
# cluster_medians = cluster_groups.median()
# cluster_counts = cluster_groups.count()

"""### Density-based (DBSCAN):

DBSCAN can work with either, but normalized data might be better if there are significant differences in scale, as DBSCAN is density-based and sensitive to the distance between points.

using df_normalized
"""

from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=5)  # Example parameters
labels_dbscan = dbscan.fit_predict(df_normalized)

plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels_dbscan)  # Replace labels_dbscan with your DBSCAN labels
plt.title("DBSCAN Clustering")
plt.show()

noise_points = sum(labels_dbscan == -1)
print("Noise Points:", noise_points)
# Further analysis can be done by examining the points in each cluster and those marked as noise

"""### Distribution-based (Gaussian Models):

Gaussian Mixture Models (a common distribution-based method) generally prefer standardized data, especially if the data is expected to conform to a normal distribution.
"""

from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=3)  # Example: 3 components
gmm.fit(df_standardized)
labels_gmm = gmm.predict(df_standardized)

plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels_gmm)  # Replace labels_gmm with your Gaussian Mixture labels
plt.title("Gaussian Mixture Clustering")
plt.show()

