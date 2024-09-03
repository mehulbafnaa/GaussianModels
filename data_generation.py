import numpy as np

def create_custom_dataset(n_samples=1000, n_features=2, n_clusters=3, cluster_std=1.0, random_state=None):
    """
    Create a custom dataset with specified number of samples, features, and clusters.

    This function generates a synthetic dataset with clustered data points. The clusters
    are created around randomly generated center points, with Gaussian noise added to
    each data point.

    Parameters:
    -----------
    n_samples : int, optional (default=1000)
        The total number of samples in the dataset.
    n_features : int, optional (default=2)
        The number of features for each sample.
    n_clusters : int, optional (default=3)
        The number of clusters to generate.
    cluster_std : float, optional (default=1.0)
        The standard deviation of the clusters.
    random_state : int or None, optional (default=None)
        Seed for the random number generator. Use for reproducibility.

    Returns:
    --------
    X : numpy.ndarray
        Array of shape (n_samples, n_features) containing the generated data points.
    centers : numpy.ndarray
        Array of shape (n_clusters, n_features) containing the cluster centers.

    Notes:
    ------
    - The function uses numpy's random number generation for creating cluster centers
      and data points.
    - Cluster centers are spread out by multiplying with a factor of 5.
    - The dataset is shuffled before being returned.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate cluster centers
    centers = np.random.randn(n_clusters, n_features) * 5  # Multiply by 5 to spread clusters
    
    # Generate data points around cluster centers
    X = np.vstack([
        np.random.randn(n_samples // n_clusters, n_features) * cluster_std + center
        for center in centers
    ])
    
    np.random.shuffle(X)  # Shuffle the dataset
    
    return X, centers