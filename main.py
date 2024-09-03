import numpy as np
from sklearn.mixture import GaussianMixture
import time

from data_generation import create_custom_dataset
from GMM import GMM
from visualization import plot_comparison
from metrics import compare_models

def main():
    # Generate sample data
    np.random.seed(42)
    X, true_centers = create_custom_dataset(n_samples=2000, n_features=2, n_clusters=4, cluster_std=1.0, random_state=42)

    # Define true labels based on the nearest center (for evaluation purposes only)
    true_labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - true_centers, axis=2), axis=1)

    # Fit custom GMM
    start_time = time.time()
    custom_gmm = GMM(n_components=4, n_iter=100)
    custom_log_likelihood = custom_gmm.fit(X)
    custom_time = time.time() - start_time
    custom_labels = custom_gmm.predict(X)

    # Fit scikit-learn GMM
    start_time = time.time()
    sklearn_gmm = GaussianMixture(n_components=4, n_init=1, random_state=42)
    sklearn_gmm.fit(X)
    sklearn_time = time.time() - start_time
    sklearn_labels = sklearn_gmm.predict(X)
    sklearn_log_likelihood = sklearn_gmm.score(X) * X.shape[0]

    # Compare models
    compare_models(X, true_labels, custom_labels, sklearn_labels, custom_time, sklearn_time, custom_log_likelihood, sklearn_log_likelihood)

    # Visualize results
    plot_comparison(X, true_labels, custom_labels, sklearn_labels)

    # Print model parameters
    print("\nCustom GMM Means:")
    print(custom_gmm.means)
    print("\nScikit-learn GMM Means:")
    print(sklearn_gmm.means_)

    print("\nCustom GMM Weights:")
    print(custom_gmm.weights)
    print("\nScikit-learn GMM Weights:")
    print(sklearn_gmm.weights_)

    print("\nTrue Cluster Centers:")
    print(true_centers)

if __name__ == "__main__":
    main()