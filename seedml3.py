# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
cols = ["area", "parameter","compactness","length","width","asymetric", "groove","class"]

# %%
df = pd.read_csv('seeds_dataset.txt', names= cols, sep="\s+")

# %%
df.head()

# %%
for i in range(len(cols)-1):
    for j in range(i+1, len(cols)-1):
        x_label = cols[i]
        y_label =cols[j]
        sns.scatterplot(x=x_label, y= y_label, data= df, hue= 'class')
        plt.show()

# %%
#Clustering

from sklearn.cluster import KMeans

# %%
x = "compactness"
y = "asymetric"

# %%

X = df[[x, y]].values

# %%
kmeans =KMeans(n_clusters =3).fit(X)

# %%
clusters= kmeans.labels_

# %%
clusters

# %%
df["class"].values

# %%
cluster_df = pd.DataFrame(np.hstack((X, clusters.reshape(-1,1))), columns =[x,y, "class"])

# %%
#kmeans classes

sns.scatterplot(x = x, y=y, hue ='class', data = cluster_df)
plt.plot()

# %%
#Higher dimensions

X = df[cols[:-1]].values

# %%
kmeans = KMeans(n_clusters = 3).fit(X)
cluster_df = pd.DataFrame(np.hstack((X, kmeans.labels_.reshape(-1, 1))), columns=df.columns)

# %%
# K Means classes
sns.scatterplot(x=x, y=y, hue='class', data=df)
plt.plot()

# %%
#PCA

from sklearn.decomposition import PCA

# %%
pca =PCA(n_components = 2)
transformed_x = pca.fit_transform(X)

# %%
X.shape

# %%
transformed_x.shape

# %%
transformed_x[:5]

# %%
plt.scatter(transformed_x[:,0], transformed_x[:,1])
plt.show()

# %%
kmeans_pca_df = pd.DataFrame(np.hstack((transformed_x, kmeans.labels_.reshape(-1, 1))), columns=["pca1", "pca2", "class"])

# %%
truth_pca_df = pd.DataFrame(np.hstack((transformed_x, df["class"].values.reshape(-1, 1))), columns=["pca1", "pca2", "class"])

# %%
# K Means classes
sns.scatterplot(x="pca1", y="pca2", hue='class', data=kmeans_pca_df)
plt.plot()

# %%
# Truth classes
sns.scatterplot(x="pca1", y="pca2", hue='class', data=truth_pca_df)
plt.plot()

# %%



