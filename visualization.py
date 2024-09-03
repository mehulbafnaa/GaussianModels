import matplotlib.pyplot as plt

def plot_clusters(X, labels, title):
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.title(title)
    plt.colorbar(scatter)
    plt.show()

def plot_comparison(X, true_labels, custom_labels, sklearn_labels):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))
    
    ax1.scatter(X[:, 0], X[:, 1], c=true_labels, cmap='viridis')
    ax1.set_title('True Clusters')
    
    ax2.scatter(X[:, 0], X[:, 1], c=custom_labels, cmap='viridis')
    ax2.set_title('Custom GMM Clusters')
    
    ax3.scatter(X[:, 0], X[:, 1], c=sklearn_labels, cmap='viridis')
    ax3.set_title('Scikit-learn GMM Clusters')
    
    plt.tight_layout()
    plt.show()