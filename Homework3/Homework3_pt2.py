import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.base import clone
from sklearn.metrics import confusion_matrix

# -------------------------------
# Helper function to run experiments
# -------------------------------
def run_classification_experiment(X, y, dr_choice, train_sizes, random_state=42):
    """
    Run classification experiments over different training sizes.
    
    Parameters:
      X          : data features (numpy array)
      y          : labels (numpy array)
      dr_choice  : string indicating the DR method to use:
                   'NoDR' for no dimensionality reduction,
                   'PCA' for PCA (with n_components=2),
                   'LDA' for LDA (with n_components=2).
      train_sizes: list of training sizes (fractions)
      random_state: random state for reproducibility
      
    Returns:
      results: dictionary with keys as training sizes. For each training size,
               a dictionary of classifier performance metrics is stored.
    """
    # Define classifiers to test
    classifiers = {
        'Naive Bayes': GaussianNB(),
        'SVM-RBF': SVC(kernel='rbf', gamma='scale'),
        'SVM-Poly': SVC(kernel='poly', degree=3, gamma='scale')
    }
    
    # Dictionary to store results: keys = training size
    results = {}
    
    for t_size in train_sizes:
        # Split dataset (using stratification)
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=t_size, 
                                                            stratify=y, random_state=random_state)
        # Standardize using training set only
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Apply dimensionality reduction if needed
        if dr_choice == 'NoDR':
            X_train_trans, X_test_trans = X_train, X_test
        elif dr_choice == 'PCA':
            dr_model = PCA(n_components=2)
            X_train_trans = dr_model.fit_transform(X_train)
            X_test_trans = dr_model.transform(X_test)
        elif dr_choice == 'LDA':
            # For LDA, we need the labels when fitting
            dr_model = LinearDiscriminantAnalysis(n_components=2)
            X_train_trans = dr_model.fit_transform(X_train, y_train)
            X_test_trans = dr_model.transform(X_test)
        else:
            raise ValueError("dr_choice must be one of: 'NoDR', 'PCA', 'LDA'.")
        
        # For this training size, store results for each classifier
        results[t_size] = {}
        for clf_name, clf in classifiers.items():
            # Use clone so that each run is independent
            clf_instance = clone(clf)
            clf_instance.fit(X_train_trans, y_train)
            train_acc = clf_instance.score(X_train_trans, y_train)
            test_acc = clf_instance.score(X_test_trans, y_test)
            
            # Store classifier and predictions for later analysis (if needed)
            results[t_size][clf_name] = {
                'train_acc': train_acc,
                'test_acc': test_acc,
                'clf': clf_instance,
                'X_train_trans': X_train_trans,
                'y_train': y_train,
                'X_test_trans': X_test_trans,
                'y_test': y_test
            }
    return results

# -------------------------------
# Function to plot accuracy vs. training size
# -------------------------------
def plot_accuracies(results_dict, dataset_name, metric='test_acc'):
    """
    Plot accuracy (training or test) vs. training size.
    
    Parameters:
      results_dict: nested dictionary with structure:
                    results_dict[dr_method][training_size][classifier][metric]
      dataset_name: string for dataset title
      metric      : 'train_acc' or 'test_acc'
    """
    # Extract training sizes from one of the DR method dictionaries.
    train_sizes = sorted(list(next(iter(results_dict.values())).keys()))
    plt.figure(figsize=(8, 5))
    for dr_method in results_dict:
        # Use the first training size to get classifier names
        first_ts = train_sizes[0]
        for clf_name in results_dict[dr_method][first_ts]:
            # For each classifier, gather the accuracy for each training size.
            accs = [results_dict[dr_method][ts][clf_name][metric] for ts in train_sizes]
            plt.plot(np.array(train_sizes)*100, accs, marker='o', linestyle='--', 
                     label=f'{dr_method} - {clf_name}')
    plt.xlabel('Training Size (%)')
    plt.ylabel('Accuracy')
    metric_title = "Training" if metric == 'train_acc' else "Test"
    plt.title(f'{dataset_name}: {metric_title} Accuracy vs Training Size')
    plt.grid(True)
    plt.legend()
    plt.show()

# -------------------------------
# Function to compute per-class accuracy given predictions and true labels
# -------------------------------
def compute_per_class_accuracy(y_true, y_pred):
    """
    Computes per-class accuracy from a confusion matrix.
    
    Returns:
      A dictionary with class labels as keys and per-class accuracy as values.
    """
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = {}
    for i in range(cm.shape[0]):
        if cm[i].sum() > 0:
            per_class_acc[i] = cm[i, i] / cm[i].sum()
        else:
            per_class_acc[i] = np.nan
    return per_class_acc

# -------------------------------
# Main function
# -------------------------------
def main():
    # Define training sizes as fractions
    train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Define DR method choices (for our experiments we want: NoDR, PCA, and LDA)
    dr_methods = ['NoDR', 'PCA', 'LDA']
    
    # Containers to store results for each dataset and DR method
    # results_all[dataset][dr_method][training_size][classifier]
    results_all = {'Iris': {}, 'Indian Pines': {}}
    
    # ================================
    #         IRIS DATASET
    # ================================
    print("Processing Iris dataset...")
    iris = load_iris()
    X_iris = iris.data
    y_iris = iris.target

    # For Iris, run experiments for each DR option
    results_all['Iris'] = {}
    for dr in dr_methods:
        print(f"Running Iris experiment with DR = {dr}")
        results_all['Iris'][dr] = run_classification_experiment(X_iris, y_iris, dr, train_sizes)
    
    # Plot training and test accuracies for Iris
    plot_accuracies(results_all['Iris'], "Iris", metric='train_acc')
    plot_accuracies(results_all['Iris'], "Iris", metric='test_acc')
    
    # ================================
    #     INDIAN PINES DATASET
    # ================================
    print("Processing Indian Pines dataset...")
    # Load Indian Pines hyperspectral data and ground truth labels
    indian_pines = scipy.io.loadmat("indianR.mat")
    indian_labels = scipy.io.loadmat("indian_gth.mat")
    
    # Print keys for reference (optional)
    print("Indian Pines keys:", indian_pines.keys())
    
    # Extract hyperspectral data and transpose so that each row is a sample
    X_pines = indian_pines['X'].T
    # Extract and flatten the ground truth labels
    y_pines = indian_labels['gth'].flatten()
    
    # Remove unclassified areas (zero labels)
    if X_pines.shape[0] == y_pines.shape[0]:
        mask = y_pines > 0
        X_pines = X_pines[mask]
        y_pines = y_pines[mask]
    else:
        print("Dimension mismatch! Check dataset preprocessing.")
    
    # Run experiments for Indian Pines for each DR option
    results_all['Indian Pines'] = {}
    for dr in dr_methods:
        print(f"Running Indian Pines experiment with DR = {dr}")
        results_all['Indian Pines'][dr] = run_classification_experiment(X_pines, y_pines, dr, train_sizes)
    
    # Plot training and test accuracies for Indian Pines
    plot_accuracies(results_all['Indian Pines'], "Indian Pines", metric='train_acc')
    plot_accuracies(results_all['Indian Pines'], "Indian Pines", metric='test_acc')
    
    # -------------------------------------------
    # Tabulate classwise accuracies for Indian Pines DR methods at 30% training size
    # (Case i: with dimensionality reduction: PCA and LDA)
    # -------------------------------------------
    print("\nClasswise classification accuracies for Indian Pines (30% training) with DR:")
    dr_tabulation = {}
    for dr in ['PCA', 'LDA']:
        # Create a DataFrame to store per-class accuracies for each classifier
        per_class_df = pd.DataFrame()
        for clf_name in results_all['Indian Pines'][dr][0.3]:
            # Get the stored info for the current classifier at training size 30%
            info = results_all['Indian Pines'][dr][0.3][clf_name]
            y_true = info['y_test']
            y_pred = info['clf'].predict(info['X_test_trans'])
            class_acc = compute_per_class_accuracy(y_true, y_pred)
            # Convert dictionary to pandas Series (index = class labels)
            per_class_series = pd.Series(class_acc, name=clf_name)
            per_class_df = pd.concat([per_class_df, per_class_series], axis=1)
        dr_tabulation[dr] = per_class_df
        print(f"\nDR Method: {dr}")
        print(per_class_df.round(3))
    
if __name__ == '__main__':
    main()
