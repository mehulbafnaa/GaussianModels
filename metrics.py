import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix

def cluster_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    return cm[row_ind, col_ind].sum() / cm.sum()

def silhouette_score(X, labels):
    n_samples = X.shape[0]
    n_clusters = len(np.unique(labels))
    
    # Compute pairwise distances efficiently
    distances = squareform(pdist(X))
    
    silhouette_vals = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Points in the same cluster as i
        a_cluster = labels == labels[i]
        a_cluster[i] = False  # Exclude the point itself
        
        if np.sum(a_cluster) > 0:
            a = np.mean(distances[i, a_cluster])
        else:
            a = 0
        
        b_values = np.zeros(n_clusters)
        
        for cluster in range(n_clusters):
            if cluster != labels[i]:
                b_cluster = labels == cluster
                if np.sum(b_cluster) > 0:
                    b_values[cluster] = np.mean(distances[i, b_cluster])
        
        b_values = b_values[b_values != 0]
        
        if len(b_values) > 0:
            b = np.min(b_values)
            silhouette_vals[i] = (b - a) / max(a, b)
    
    return np.mean(silhouette_vals)

def calinski_harabasz_score(X, labels):
    n_samples, n_features = X.shape
    n_clusters = len(np.unique(labels))
    
    if n_clusters == 1:
        return float('inf')
    
    extra_disp, intra_disp = 0., 0.
    mean = np.mean(X, axis=0)
    
    for k in range(n_clusters):
        cluster_k = X[labels == k]
        mean_k = np.mean(cluster_k, axis=0)
        extra_disp += len(cluster_k) * np.sum((mean_k - mean) ** 2)
        intra_disp += np.sum((cluster_k - mean_k) ** 2)
    
    return (
        extra_disp * (n_samples - n_clusters)
        / (intra_disp * (n_clusters - 1))
    )

def evaluate_clustering(X, labels, true_labels=None):
    silhouette = silhouette_score(X, labels)
    ch_score = calinski_harabasz_score(X, labels)
    
    results = {
        'silhouette_score': silhouette,
        'calinski_harabasz_score': ch_score
    }
    
    if true_labels is not None:
        accuracy = cluster_accuracy(true_labels, labels)
        results['accuracy'] = accuracy
    
    return results

def compare_models(X, true_labels, custom_labels, sklearn_labels, custom_time, sklearn_time, custom_log_likelihood, sklearn_log_likelihood):
    custom_metrics = evaluate_clustering(X, custom_labels, true_labels)
    sklearn_metrics = evaluate_clustering(X, sklearn_labels, true_labels)
    
    print(f"Custom GMM Time: {custom_time:.4f} seconds")
    print(f"Scikit-learn GMM Time: {sklearn_time:.4f} seconds")
    print(f"Custom GMM Silhouette Score: {custom_metrics['silhouette_score']:.4f}")
    print(f"Scikit-learn GMM Silhouette Score: {sklearn_metrics['silhouette_score']:.4f}")
    print(f"Custom GMM Calinski-Harabasz Score: {custom_metrics['calinski_harabasz_score']:.4f}")
    print(f"Scikit-learn GMM Calinski-Harabasz Score: {sklearn_metrics['calinski_harabasz_score']:.4f}")
    print(f"Custom GMM Log-Likelihood: {custom_log_likelihood:.4f}")
    print(f"Scikit-learn GMM Log-Likelihood: {sklearn_log_likelihood:.4f}")
    if 'accuracy' in custom_metrics:
        print(f"Custom GMM Accuracy (for reference): {custom_metrics['accuracy']:.4f}")
        print(f"Scikit-learn GMM Accuracy (for reference): {sklearn_metrics['accuracy']:.4f}")