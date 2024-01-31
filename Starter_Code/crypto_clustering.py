# %%
# Import required libraries and dependencies
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# %%
# Load the data into a Pandas DataFrame and make the index the "coin_id" column.
market_data_df = pd.read_csv("Resources/crypto_market_data.csv", index_col="coin_id")

# Display sample data
market_data_df.head(10)

# %%
# Generate summary statistics
market_data_df.describe()

# %% [markdown]
# ### Prepare the Data

# %%
# Use the `StandardScaler()` module from scikit-learn to normalize the data from the CSV file
scaler = StandardScaler()
scaled_data = scaler.fit_transform(market_data_df)


# %%
# Create a DataFrame with the scaled data
# Copy the crypto names from the original data
df_scaled_data = pd.DataFrame(scaled_data, columns=market_data_df.columns) 

# Set the coinid column as index
df_scaled_data["coin_id"] = market_data_df.index
df_scaled_data.set_index("coin_id", inplace=True)

# Display sample data
df_scaled_data.head()

# %% [markdown]
# ### I.  Find the Best Value for k Using the Original Scaled DataFrame. (15 pts)

# %%
# Create a list with the number of k-values to try
# Use a range from 1 to 11
k = list(range(1,11))

# Create an empty list to store the inertia values
inertia = []

# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. Create a KMeans model using the loop counter for the n_clusters
# 2. Fit the model to the data using the scaled DataFrame
# 3. Append the model.inertia_ to the inertia list

for i in k:
    model = KMeans(n_clusters=i, n_init='auto', random_state=1)
    model.fit(df_scaled_data)
    inertia.append(model.inertia_)

# Create a dictionary with the data to plot the Elbow curve
elbow_data = {
    "k": k,
    "inertia": inertia
}

# Create a DataFrame with the data to plot the elbow curve
df_elbow = pd.DataFrame(elbow_data) 

# Display the DataFrame
df_elbow

# %%
# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k. (5 pts)
df_elbow.plot.line(x="k",
                   y="inertia",
                   title="Elbow Curve",
                   xticks=k)

## I added this to keep in the practice of also using percentages to visually see breakouts
k = elbow_data["k"]
inertia = elbow_data["inertia"]
for i in range(1, len(k)):
    percentage_decrease = (inertia[i-1] - inertia[i]) / inertia[i-1] * 100
    print(f"Percentage decrease from k={k[i-1]} to k={k[i]}: {percentage_decrease:.2f}%")


# %% [markdown]
# #### Answer the following question: (5 pts)
# **Question:** What is the best value for `k`?
# 
# **Answer:** From the line plot, it appears that 4 is the best number of clusters.

# %% [markdown]
# ### II. Cluster Cryptocurrencies with K-means Using the Original Scaled Data. (10pts)

# %%
# Initialize the K-Means model using the best value for k
model = KMeans(n_clusters=4, n_init='auto', random_state=1)

# %%
# Fit the K-Means model using the scaled data
model.fit(df_scaled_data)

# %%
# Predict the clusters to group the cryptocurrencies using the scaled data
crypto_cluster_predictions = model.predict(df_scaled_data)

# View the resulting array of cluster values.
crypto_cluster_predictions

# %%
# Create a copy of the <which?> DataFrame ##(homework says original dataframe, examples show copy of the scaled datafarme)
crypto_cluster_predictions_df = df_scaled_data.copy()
crypto_cluster_predictions_df.head()

# %%
# Add a new column to the DataFrame with the predicted clusters
crypto_cluster_predictions_df["crypto_cluster"] = crypto_cluster_predictions

# Display sample data
crypto_cluster_predictions_df.head()

# %%
# Create a scatter plot using Pandas plot by setting 
# `x="price_change_percentage_24h"` and `y="price_change_percentage_7d"`.
# Use "rainbow" for the color to better visualize the data.
crypto_cluster_predictions_df.plot.scatter(
    x="price_change_percentage_24h",
    y="price_change_percentage_7d",
    c="crypto_cluster",
    colormap='rainbow')


# %% [markdown]
# ### III. Optimize Clusters with Principal Component Analysis. (10pts)

# %%
# Create a PCA model instance and set `n_components=3`.
pca = PCA(n_components=3)

# %%
# Use the PCA model with `fit_transform` on the original scaled DataFrame to reduce to three principal components.
stocks_pca_data = pca.fit_transform(df_scaled_data)

# View the first five rows of the DataFrame. 
stocks_pca_data[:5]

# %%
# Retrieve the explained variance to determine how much information  can be attributed to each principal component.
pca.explained_variance_ratio_

# %% [markdown]
# #### Answer the following question: 
# 
# **Question:** What is the total explained variance of the three principal components?
# 
# **Answer:** 0.894 is the total explained variance using 3 decimal points.

# %%
# Create a DataFrame with the PCA data.
# Copy the crypto names from the original data.
crypto_pca_data = pd.DataFrame(stocks_pca_data, columns=["PCA1","PCA2","PCA3"])

# Set the coinid column as index
crypto_pca_data["coin_id"] = df_scaled_data.index
crypto_pca_data = crypto_pca_data.set_index("coin_id")

# Display sample data
crypto_pca_data.head(10)


# %% [markdown]
# ### Find the Best Value for k Using the PCA Data (10pts)

# %%
# Create a list with the number of k-values to try
# Use a range from 1 to 11
k = list(range(1,11))


inertia = []

# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. Create a KMeans model using the loop counter for the n_clusters
# 2. Fit the model to the data using the scaled DataFrame
# 3. Append the model.inertia_ to the inertia list

for i in k:
    model = KMeans(n_clusters=i, n_init='auto', random_state=1)
    model.fit(crypto_pca_data)
    inertia.append(model.inertia_)

# Create a dictionary with the data to plot the Elbow curve
elbow_data = {
    "k": k,
    "inertia": inertia
}

# Create a DataFrame with the data to plot the elbow curve
df_elbow = pd.DataFrame(elbow_data) 

# Display the DataFrame
df_elbow

# %%
# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k. (5 pts)
df_elbow.plot.line(x="k",
                   y="inertia",
                   title="Elbow Curve",
                   xticks=k)

## I added this to keep in the practice of also using percentages to visually see breakouts
k = elbow_data["k"]
inertia = elbow_data["inertia"]
for i in range(1, len(k)):
    percentage_decrease = (inertia[i-1] - inertia[i]) / inertia[i-1] * 100
    print(f"Percentage decrease from k={k[i-1]} to k={k[i]}: {percentage_decrease:.2f}%")

# %% [markdown]
# #### Answer the following questions: 
# * **Question:** What is the best value for `k` when using the PCA data?
# 
#   * **Answer:**  4 appears to be the best value for 'k' when using the PCA data.
# 
# 
# * **Question:** Does it differ from the best k value found using the original data?
# 
#   * **Answer:** This does not differ from the best k value found using the original data.

# %% [markdown]
# ### Cluster Cryptocurrencies with K-means Using the PCA Data (10pts)

# %%
# Initialize the K-Means model using the best value for k
model = KMeans(n_clusters=4, n_init='auto', random_state=1)

# %%
# Fit the K-Means model using the PCA data
model.fit(crypto_pca_data)

# %%
# Predict the clusters to group the cryptocurrencies using the PCA data
stock_clusters = model.predict(crypto_pca_data)

# Print the resulting array of cluster values.
stock_clusters

# %%
# Create a copy of the DataFrame with the PCA data
stock_clusters_pred = crypto_pca_data.copy()


# Add a new column to the DataFrame with the predicted clusters
stock_clusters_pred["crypto_cluster"] = stock_clusters

# Display sample data
stock_clusters_pred.head()

# %%
# Create a scatter plot using hvPlot by setting `x="PCA1"` and `y="PCA2"`. 
stock_clusters_pred.plot.scatter(
    x="PCA1",
    y="PCA2",
    c="crypto_cluster",
    colormap='rainbow') #switched to rainbow as I can't differentiate the blues/ greens very well

# %% [markdown]
# ### Determine the Weights of Each Feature on each Principal Component (15pts)

# %%
# Use the columns from the original scaled DataFrame as the index.
pca_component_weights = pd.DataFrame(pca.components_.T, columns=['PCA1','PCA2','PCA3'],index = df_scaled_data.columns)
pca_component_weights.head()

# %% [markdown]
# #### Answer the following question: 
# 
# * **Question:** Which features have the strongest positive or negative influence on each component? 
#  
# * **Answer:** 
# 24h price change, PCA1 has the strongest influence with PCA3 having the least.
#  7d price change, PCA3 has the strongest influence with PCA1 having the least.
# 14d price change, PCA2 has the strongest influence with PCA1 having the least.
# 30d price change, PCA2 has the strongest influence with PCA3 having the least.
# 60d price change, PCA2 has the strongest influence with PCA3 having the least.  


