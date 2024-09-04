import numpy as np
from sklearn.mixture import GaussianMixture
import time
import matplotlib.pyplot as plt

from data_generation import create_custom_dataset
from gmm_model import GMM
from visualization import plot_comparison
from metrics import compare_models, evaluate_clustering

def fit_and_evaluate_gmm(gmm, X, true_labels, name):
    start_time = time.time()
    gmm.fit(X)
    fit_time = time.time() - start_time

    labels = gmm.predict(X)
    
    if hasattr(gmm, 'compute_log_likelihood'):
        log_likelihood = gmm.compute_log_likelihood(X)
    else:
        log_likelihood = gmm.score(X) * X.shape[0]

    metrics = evaluate_clustering(X, labels, true_labels)
    metrics['fit_time'] = fit_time
    metrics['log_likelihood'] = log_likelihood

    if hasattr(gmm, 'bic'):
        metrics['bic'] = gmm.bic(X)
    if hasattr(gmm, 'aic'):
        metrics['aic'] = gmm.aic(X)

    print(f"\n{name} GMM Results:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    return gmm, labels, metrics

def print_model_parameters(custom_gmm, sklearn_gmm, true_centers):
    print("\nModel Parameters:")
    print("Custom GMM Means:")
    print(custom_gmm.means)
    print("\nScikit-learn GMM Means:")
    print(sklearn_gmm.means_)
    print("\nCustom GMM Weights:")
    print(custom_gmm.weights)
    print("\nScikit-learn GMM Weights:")
    print(sklearn_gmm.weights_)
    print("\nTrue Cluster Centers:")
    print(true_centers)

def main():
    # Generate sample data
    np.random.seed(42)
    X, true_centers = create_custom_dataset(n_samples=2000, n_features=2, n_clusters=4, cluster_std=1.0, random_state=42)

    # Define true labels based on the nearest center (for evaluation purposes only)
    true_labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - true_centers, axis=2), axis=1)

    # Fit and evaluate custom GMM
    custom_gmm = GMM(n_components=4, n_iter=100, n_init=5, random_state=42)
    custom_gmm, custom_labels, custom_metrics = fit_and_evaluate_gmm(custom_gmm, X, true_labels, "Custom")

    # Fit and evaluate scikit-learn GMM
    sklearn_gmm = GaussianMixture(n_components=4, n_init=5, random_state=42)
    sklearn_gmm, sklearn_labels, sklearn_metrics = fit_and_evaluate_gmm(sklearn_gmm, X, true_labels, "Scikit-learn")

    # Compare models
    compare_models(X, true_labels, custom_labels, sklearn_labels, 
                   custom_metrics['fit_time'], sklearn_metrics['fit_time'],
                   custom_metrics['log_likelihood'], sklearn_metrics['log_likelihood'],
                   custom_model=custom_gmm, sklearn_model=sklearn_gmm)

    # Visualize results
    plot_comparison(X, true_labels, custom_labels, sklearn_labels)

    # Print model parameters
    print_model_parameters(custom_gmm, sklearn_gmm, true_centers)

    # Optional: Add model selection using BIC/AIC
    n_components_range = range(1, 10)
    bic_scores = [GMM(n_components=n, random_state=42).fit(X).bic(X) for n in n_components_range]
    aic_scores = [GMM(n_components=n, random_state=42).fit(X).aic(X) for n in n_components_range]

    plt.figure(figsize=(10, 6))
    plt.plot(n_components_range, bic_scores, label='BIC')
    plt.plot(n_components_range, aic_scores, label='AIC')
    plt.xlabel('Number of components')
    plt.ylabel('Score')
    plt.title('Model Selection Scores')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()