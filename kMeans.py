import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import plotly.express as px
import kaleido  # For exporting images if needed

# -----------------------------
# Load the imputed, z-score standardized DataFrame.
# -----------------------------
df = pd.read_pickle("static_dataframe_imputed_zscore.pkl")

# -----------------------------
# Define the features for clustering.
# -----------------------------
features = [
    "sy_snum",
    "pl_controv_flag",
    "pl_orbper",
    "pl_orbsmax",
    "pl_rade",
    "pl_radj",
    "pl_bmasse",
    "pl_bmassj",
    "pl_orbeccen",
    "pl_eqt",
    "ttv_flag",
    "st_teff",
    "st_rad",
    "st_mass",
    "st_met",
    "st_logg",
    "ra",
    "dec",
    "sy_dist",
    "sy_vmag",
    "sy_kmag",
    "sy_gaiamag"
]

# -----------------------------
# Prepare the data for clustering.
# -----------------------------
df_cluster = df.dropna(subset=features).copy()
X = df_cluster[features]

# -----------------------------
# Run KMeans clustering.
# -----------------------------
# For example, using X clusters.
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)
df_cluster['cluster'] = kmeans.labels_

# -----------------------------
# Compute the R²-like metric.
# -----------------------------
# SSW is the within-cluster sum of squares (inertia)
SSW = kmeans.inertia_
# Total sum of squares: sum of squared distances from each point to the overall mean.
SS_T = np.sum((X - np.mean(X, axis=0))**2)
R2 = 1 - (SSW / SS_T)
print("Explained Variance (R²-like metric):", R2)

# -----------------------------
# Compute Cluster Summary Statistics.
# -----------------------------
cluster_summary = df_cluster.groupby('cluster')[features].mean()
print("Cluster Summary (Mean values):")
print(cluster_summary)

# -----------------------------
# Export the cluster summary statistics to a CSV file.
# -----------------------------
output_filename = "cluster_summary.csv"
cluster_summary.to_csv(output_filename)
print(f"\nCluster summary statistics have been saved to {output_filename}.")

# -----------------------------
# Evaluate Clustering: Elbow Curve and Silhouette Scores.
# -----------------------------
k_values = range(2, 11)  # Evaluate k from 2 to 10
inertias = []
silhouette_scores = []

for k_val in k_values:
    kmeans_test = KMeans(n_clusters=k_val, random_state=42)
    kmeans_test.fit(X)
    inertias.append(kmeans_test.inertia_)
    sil_score = silhouette_score(X, kmeans_test.labels_)
    silhouette_scores.append(sil_score)

# Plot the elbow curve.
plt.figure(figsize=(8, 4))
plt.plot(k_values, inertias, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (WCSS)")
plt.title("Elbow Curve for KMeans Clustering")
plt.grid(True)
plt.show()

# Plot silhouette scores.
plt.figure(figsize=(8, 4))
plt.plot(k_values, silhouette_scores, marker='o', color='green')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Average Silhouette Score")
plt.title("Silhouette Scores for Different k")
plt.grid(True)
plt.show()

# -----------------------------
# 3D Visualization using PCA and Plotly.
# -----------------------------
df_cluster["cluster"] = df_cluster["cluster"].astype(str)

from sklearn.decomposition import PCA
import plotly.express as px


pca = PCA(n_components=3, random_state=42)
X_pca = pca.fit_transform(X)

# Create a DataFrame of the PCA loadings (optional; for interpretation)
loadings_df = pd.DataFrame(pca.components_,
                           columns=features,
                           index=["PC1", "PC2", "PC3"])
print("PCA Loadings:")
print(loadings_df)

print("\nExplained Variance Ratio:")
for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1}: {ratio:.4f}")

# Add principal components to the DataFrame
df_cluster["PC1"] = X_pca[:, 0]
df_cluster["PC2"] = X_pca[:, 1]
df_cluster["PC3"] = X_pca[:, 2]

# Convert cluster column to string to ensure discrete coloring
df_cluster["cluster"] = df_cluster["cluster"].astype(str)

# Create a new column for marker size based on 'st_mass' ensuring all values are positive.
df_cluster["marker_size"] = df_cluster["st_mass"] - df_cluster["st_mass"].min() + 1

# Create the 3D scatter plot with Plotly Express
fig = px.scatter_3d(
    df_cluster,
    x="PC1",
    y="PC2",
    z="PC3",
    color="cluster",
    title="3D Visualization of Clusters (PCA Reduced)",
    labels={
        'PC1': 'Principal Component 1',
        'PC2': 'Principal Component 2',
        'PC3': 'Principal Component 3',
        'cluster': 'Cluster'
    },
    size="marker_size",         # Use the adjusted marker_size column derived from st_mass
    size_max=25,                # Maximum marker size
    color_discrete_sequence=["red", "blue", "orange", "black"]
)
fig.show()