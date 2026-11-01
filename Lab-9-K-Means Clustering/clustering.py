
print("\nSample records with cluster labels:")
display(df_kmeans.head())

# --- Code cell 19 ---
hierarchical_model = AgglomerativeClustering(
    n_clusters=K,linkage="ward",metric="euclidean"
)

hierarchical_labels = hierarchical_model.fit_predict(X_scaled)

# --- Code cell 20 ---
df_hierarchical = df.copy()
df_hierarchical["Hierarchical_Cluster"] = hierarchical_labels

print("Hierarchical clustering completed.")
print("\nNumber of customers in each cluster:")
print(
    df_hierarchical["Hierarchical_Cluster"].value_counts().sort_index()
)

print("\nSample records with cluster labels:")
display(df_hierarchical.head())

# --- Code cell 21 ---
Z = linkage(X_scaled, method="ward",metric="euclidean")

plt.figure(figsize=(16,7))
dendrogram(Z,leaf_rotation=90,leaf_font_size=6)

plt.title("Dendogram - Hierarchical Clustering")
plt.xlabel("Customer Index")
plt.ylabel("Ward Linkage Distance")
plt.tight_layout()
plt.show()

# --- Code cell 22 ---
plt.figure(figsize=(12,6))
dendrogram(Z, truncate_mode="lastp",p=30,show_leaf_counts=True,leaf_rotation=45)

plt.title("Truncated Dendrogram - Last 30 Merged Groups")
plt.xlabel("Cluster or Group")
plt.ylabel("Ward Linkage Distance")
plt.tight_layout()
plt.show()

# --- Code cell 23 ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Variance explained by the two PCA components:",round(pca.explained_variance_ratio_.sum()*100,2),"%")

plot_df = pd.DataFrame({"PC1": X_pca[:, 0],"PC2": X_pca[:,1],
                        "KMeans_Cluster" : kmeans_labels.astype(str),
                        "Hierarchical_Cluster" : hierarchical_labels.astype(str)
                        })

# --- Code cell 24 ---
plt.figure(figsize=(9,6))
sns.scatterplot(
    data=plot_df,x="PC1",y="PC2",hue="KMeans_Cluster",palette="Set1",s=70)

plt.title("Customer Segments - K-Means Clustering")

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")

plt.tight_layout()
plt.show()

# --- Code cell 25 ---
def evaluate_clustering(X_data, labels, model_name):
  return{
      "Algorithm": model_name,
      "Number of Clusters" : len(np.unique(labels)),
      "Silhouette Score" : silhouette_score(X_data, labels),
      "Davies-Bouldin Score" : davies_bouldin_score(X_data, labels),
      "Calinski-Harabasz Score" : calinski_harabasz_score(X_data, labels)
  }

# --- Code cell 26 ---
cluster_sizes = pd.DataFrame({
    "K-Means" : pd.Series(kmeans_labels).value_counts().sort_index(),
    "Hierarchical" : pd.Series(hierarchical_labels).value_counts().sort_index()
}).fillna(0).astype(int)

print("Cluster sizes:")
display(cluster_sizes)

# --- Code cell 27 ---
kmeans_profile = df[features].copy()
kmeans_profile["Cluster"] = kmeans_labels

kmeans_means = kmeans_profile.groupby("Cluster")[features].mean().round(2)

print("Avearge spending profile of K-Means clusters:")
display(kmeans_means)

# --- Code cell 28 ---
cluster_distribution = pd.DataFrame({
    "Customer Count": pd.Series(kmeans_labels).value_counts().sort_index()
})

cluster_distribution["Percentage"] = (
    cluster_distribution["Customer Count"]/len(df) * 100
 ).round(2)
print("K-Means customer distribution:")
display(cluster_distribution)

# --- Code cell 29 ---
plt.figure(figsize=(12,6))
sns.heatmap(kmeans_means,annot=True,fmt=".0f",cmap="YlGnBu")

plt.title("Average Spending by K-Means Cluster")
plt.xlabel("Spending Category")
plt.ylabel("Cluster")
plt.tight_layout()
plt.show()

# --- Code cell 30 ---
channel_mix = pd.crosstab(df["Channel"],kmeans_labels,normalize="columns") * 100

channel_mix.columns = [f"Cluster {c}" for c in channel_mix.columns]

print("Channel composition within each K-Means cluster (%):")

display(channel_mix.round(2))
