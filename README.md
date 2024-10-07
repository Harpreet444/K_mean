# Iris Flower Clustering using K-Means

This project forms clusters of Iris flowers using petal width and petal length features from the Iris dataset. The goal is to find the optimal number of clusters using the elbow method.

## Requirements

- scikit-learn
- matplotlib
- pandas
- seaborn

## Dataset

The dataset used is the `Iris` dataset from `sklearn.datasets`, which contains measurements of iris flowers.

## Steps

1. **Load Dataset**: Load the iris dataset using `load_iris()` from `sklearn.datasets`.
2. **Data Preparation**: Drop sepal length and sepal width features for simplicity.
3. **Scaling**: Apply MinMax scaling to the petal length and petal width features to standardize the data.
4. **Elbow Plot**: Draw an elbow plot to determine the optimal value of K.
5. **Clustering**: Perform K-Means clustering with the optimal number of clusters.
6. **Visualization**: Plot the clusters using a scatter plot.

## Code

```python
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Load dataset
data_set = load_iris()

# Create DataFrame
df = pd.DataFrame(data_set.data, columns=data_set.feature_names)

# Drop sepal length and sepal width columns
df = df.drop(['sepal length (cm)', 'sepal width (cm)'], axis='columns')

# Add target column
df['target'] = data_set.target

# Apply scaling
scaler = MinMaxScaler()
df[['petal length (cm)', 'petal width (cm)']] = scaler.fit_transform(df[['petal length (cm)', 'petal width (cm)']])

# Plot data points
df0 = df[df.target == 0]
df1 = df[df.target == 1]
df2 = df[df.target == 2]

plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'], color='g', marker='*', label=data_set.target_names[0])
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'], color='r', marker='+', label=data_set.target_names[1])
plt.scatter(df2['petal length (cm)'], df2['petal width (cm)'], color='b', marker='o', label=data_set.target_names[2])

plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.legend()
plt.show()

# Elbow method to determine optimal K
sse = []
for i in range(1, 9):
    km = KMeans(n_clusters=i)
    km.fit(df[['petal length (cm)', 'petal width (cm)']])
    sse.append(km.inertia_)
    
plt.plot(range(1, 9), sse)
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.title('Elbow Plot')
plt.show()

# Perform K-Means with optimal K
km = KMeans(n_clusters=3)
df['cluster'] = km.fit_predict(df[['petal length (cm)', 'petal width (cm)']])

# Plot clusters
plt.scatter(df['petal length (cm)'], df['petal width (cm)'], c=df['cluster'], cmap='viridis')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('K-Means Clustering')
plt.show()
