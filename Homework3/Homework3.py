# Homework 3 
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load the Indian Pines dataset
indian_pines = scipy.io.loadmat("indianR.mat")  # Load Hyperspectral Data

# Print available keys in the loaded .mat file
print(indian_pines.keys())

# Load Iris dataset
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

# Standardizing the data
scaler = StandardScaler()
X_iris_scaled = scaler.fit_transform(X_iris)

# Apply PCA to the Iris dataset
pca_iris = PCA()
X_iris_pca = pca_iris.fit_transform(X_iris_scaled)

# Plot explained variance for Iris dataset
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(pca_iris.explained_variance_ratio_) + 1), 
         np.cumsum(pca_iris.explained_variance_ratio_), marker='o', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance of PCA - Iris Dataset')
plt.grid()
plt.show()

# Reduce Iris data to 2D
pca_iris_2D = PCA(n_components=2)
X_iris_2D = pca_iris_2D.fit_transform(X_iris_scaled)

# Plot the transformed data (Iris)
plt.figure(figsize=(8, 5))
plt.scatter(X_iris_2D[:, 0], X_iris_2D[:, 1], c=y_iris, cmap='viridis', edgecolors='k', alpha=0.8)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA 2D Visualization - Iris Dataset')
plt.colorbar(label='Classes')
plt.grid()
plt.show()

# Load the Indian Pines dataset using the correct keys
indian_pines = scipy.io.loadmat("indianR.mat")
indian_labels = scipy.io.loadmat("indian_gth.mat")

# Extract the hyperspectral data and transpose it so that each row is a pixel/sample
X_pines = indian_pines['X'].T

# Extract and flatten labels
y_pines = indian_labels['gth'].flatten()

# Verify dimensions
print(f"X_pines shape: {X_pines.shape}")  # Should now be (21025, 202)
print(f"y_pines shape: {y_pines.shape}")  # Should be (21025,)

# Ensure matching dimensions before applying the mask
if X_pines.shape[0] == y_pines.shape[0]:
    # Remove zero labels (Unclassified areas)
    mask = y_pines > 0
    X_pines = X_pines[mask]
    y_pines = y_pines[mask]
else:
    print("Dimension mismatch! Check dataset preprocessing.")

# Standardizing the data
X_pines_scaled = scaler.fit_transform(X_pines)

# Apply PCA to Indian Pines dataset
pca_pines = PCA()
X_pines_pca = pca_pines.fit_transform(X_pines_scaled)

# Plot explained variance for Indian Pines dataset
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(pca_pines.explained_variance_ratio_) + 1), 
         np.cumsum(pca_pines.explained_variance_ratio_), marker='o', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance of PCA - Indian Pines Dataset')
plt.grid()
plt.show()

# Reduce Indian Pines data to 2D
pca_pines_2D = PCA(n_components=2)
X_pines_2D = pca_pines_2D.fit_transform(X_pines_scaled)

# Plot the transformed data (Indian Pines)
plt.figure(figsize=(8, 5))
plt.scatter(X_pines_2D[:, 0], X_pines_2D[:, 1], c=y_pines, cmap='jet', edgecolors='k', alpha=0.6)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA 2D Visualization - Indian Pines Dataset')
plt.colorbar(label='Classes')
plt.grid()
plt.show()
