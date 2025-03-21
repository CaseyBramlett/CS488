import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# ================================
#           IRIS DATASET
# ================================
print("Processing Iris dataset...")

# Load Iris dataset
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

# Standardize the Iris data
scaler = StandardScaler()
X_iris_scaled = scaler.fit_transform(X_iris)

# -------------------------------------------------
# i) PCA: Plot the explained variance for all PCs
# -------------------------------------------------
pca_iris = PCA()
X_iris_pca_all = pca_iris.fit_transform(X_iris_scaled)
explained_variance_ratio_iris = pca_iris.explained_variance_ratio_

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance_ratio_iris) + 1),
         np.cumsum(explained_variance_ratio_iris),
         marker='o', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Iris Dataset: PCA Explained Variance')
plt.grid(True)
plt.show()

# -------------------------------------------------
# ii) PCA: Reduce data to 2 dimensions and visualize
# -------------------------------------------------
pca_iris_2D = PCA(n_components=2)
X_iris_pca_2D = pca_iris_2D.fit_transform(X_iris_scaled)

plt.figure(figsize=(8, 5))
plt.scatter(X_iris_pca_2D[:, 0], X_iris_pca_2D[:, 1],
            c=y_iris, cmap='viridis', edgecolors='k', alpha=0.8)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Iris Dataset: PCA 2D Visualization')
plt.colorbar(label='Class Label')
plt.grid(True)
plt.show()

# -------------------------------------------------
# iii) LDA: Reduce data to 2 dimensions and visualize
# -------------------------------------------------
lda_iris = LinearDiscriminantAnalysis(n_components=2)
X_iris_lda_2D = lda_iris.fit_transform(X_iris_scaled, y_iris)

plt.figure(figsize=(8, 5))
plt.scatter(X_iris_lda_2D[:, 0], X_iris_lda_2D[:, 1],
            c=y_iris, cmap='viridis', edgecolors='k', alpha=0.8)
plt.xlabel('Linear Discriminant 1')
plt.ylabel('Linear Discriminant 2')
plt.title('Iris Dataset: LDA 2D Visualization')
plt.colorbar(label='Class Label')
plt.grid(True)
plt.show()

# ================================
#       INDIAN PINES DATASET
# ================================
print("Processing Indian Pines dataset...")

# Load the Indian Pines dataset and ground truth labels
indian_pines = scipy.io.loadmat("indianR.mat")
indian_labels = scipy.io.loadmat("indian_gth.mat")

# (Optional) Print keys to check the structure of the .mat file
print("Indian Pines keys:", indian_pines.keys())

# Extract the hyperspectral data and transpose so that each row is a sample (pixel)
X_pines = indian_pines['X'].T

# Extract and flatten the ground truth labels
y_pines = indian_labels['gth'].flatten()

# Verify dimensions
print("X_pines shape:", X_pines.shape)  # Expected: (number of pixels, number of spectral bands)
print("y_pines shape:", y_pines.shape)  # Expected: (number of pixels,)

# Remove zero labels (unclassified areas) if dimensions match
if X_pines.shape[0] == y_pines.shape[0]:
    mask = y_pines > 0
    X_pines = X_pines[mask]
    y_pines = y_pines[mask]
else:
    print("Dimension mismatch! Check dataset preprocessing.")

# Standardize the Indian Pines data
X_pines_scaled = scaler.fit_transform(X_pines)

# -------------------------------------------------
# i) PCA: Plot the explained variance for all PCs
# -------------------------------------------------
pca_pines = PCA()
X_pines_pca_all = pca_pines.fit_transform(X_pines_scaled)
explained_variance_ratio_pines = pca_pines.explained_variance_ratio_

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance_ratio_pines) + 1),
         np.cumsum(explained_variance_ratio_pines),
         marker='o', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Indian Pines Dataset: PCA Explained Variance')
plt.grid(True)
plt.show()

# -------------------------------------------------
# ii) PCA: Reduce data to 2 dimensions and visualize
# -------------------------------------------------
pca_pines_2D = PCA(n_components=2)
X_pines_pca_2D = pca_pines_2D.fit_transform(X_pines_scaled)

plt.figure(figsize=(8, 5))
plt.scatter(X_pines_pca_2D[:, 0], X_pines_pca_2D[:, 1],
            c=y_pines, cmap='jet', edgecolors='k', alpha=0.6)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Indian Pines Dataset: PCA 2D Visualization')
plt.colorbar(label='Class Label')
plt.grid(True)
plt.show()

# -------------------------------------------------
# iii) LDA: Reduce data to 2 dimensions and visualize
# -------------------------------------------------
lda_pines = LinearDiscriminantAnalysis(n_components=2)
X_pines_lda_2D = lda_pines.fit_transform(X_pines_scaled, y_pines)

plt.figure(figsize=(8, 5))
plt.scatter(X_pines_lda_2D[:, 0], X_pines_lda_2D[:, 1],
            c=y_pines, cmap='jet', edgecolors='k', alpha=0.6)
plt.xlabel('Linear Discriminant 1')
plt.ylabel('Linear Discriminant 2')
plt.title('Indian Pines Dataset: LDA 2D Visualization')
plt.colorbar(label='Class Label')
plt.grid(True)
plt.show()
