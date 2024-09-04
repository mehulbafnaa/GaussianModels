import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture

def plot_clusters(X, labels, title, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    return scatter

def plot_contours(X, gmm, ax):
    x = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
    y = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100)
    X_, Y_ = np.meshgrid(x, y)
    XX = np.array([X_.ravel(), Y_.ravel()]).T
    Z = -gmm.score_samples(XX)
    Z = Z.reshape(X_.shape)
    ax.contour(X_, Y_, Z, levels=np.logspace(0, 2, 20), cmap='inferno', alpha=0.5)

def plot_comparison(X, true_labels, custom_gmm, sklearn_gmm):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))

    scatter1 = plot_clusters(X, true_labels, 'True Clusters', ax1)
    scatter2 = plot_clusters(X, custom_gmm.predict(X), 'Custom GMM Clusters', ax2)
    scatter3 = plot_clusters(X, sklearn_gmm.predict(X), 'Scikit-learn GMM Clusters', ax3)

    plot_contours(X, custom_gmm, ax2)
    plot_contours(X, sklearn_gmm, ax3)

    for ax in (ax1, ax2, ax3):
        ax.set_xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
        ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)

    fig.colorbar(scatter1, ax=ax1, label='Cluster')
    fig.colorbar(scatter2, ax=ax2, label='Cluster')
    fig.colorbar(scatter3, ax=ax3, label='Cluster')

    plt.tight_layout()
    plt.show()

def plot_model_selection(n_components_range, bic_scores, aic_scores):
    plt.figure(figsize=(12, 6))
    plt.plot(n_components_range, bic_scores, marker='o', label='BIC')
    plt.plot(n_components_range, aic_scores, marker='s', label='AIC')
    plt.xlabel('Number of components', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Model Selection Scores', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_log_likelihood_evolution(log_likelihood_history):
    plt.figure(figsize=(12, 6))
    plt.plot(log_likelihood_history, marker='o')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Log-likelihood', fontsize=12)
    plt.title('Log-likelihood Evolution During EM Algorithm', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()