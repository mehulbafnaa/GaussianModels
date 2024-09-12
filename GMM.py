import numpy as np
from scipy.special import logsumexp

class GMM:
    """
    Gaussian Mixture Model (GMM) implementation.

    This class implements the GMM algorithm for clustering and density estimation.

    Parameters:
    -----------
    n_components : int
        The number of mixture components.
    covariance_type : str, optional (default='full')
        The type of covariance parameters to use. 
        Must be one of 'full', 'tied', 'diag', 'spherical'.
    n_iter : int, optional (default=100)
        The number of EM iterations to perform.
    tol : float, optional (default=1e-5)
        The convergence threshold. EM iterations will stop when the lower bound 
        average gain is below this threshold.
    reg_covar : float, optional (default=1e-6)
        Non-negative regularization added to the diagonal of covariance.
    random_state : int, optional (default=None)
        Controls the random seed given to the method for initialization.
    """

    def __init__(self, n_components, covariance_type='full', n_iter=100, tol=1e-5, reg_covar=1e-6, random_state=None):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state

    def initialize_parameters(self, X):
        """
        Initialize the model parameters.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data.
        """
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)

        self.weights = np.full(self.n_components, 1/self.n_components)
        self.means = X[rng.choice(n_samples, self.n_components, replace=False)]

        if self.covariance_type == 'full':
            self.covariances = np.stack([np.eye(n_features) for _ in range(self.n_components)])
        elif self.covariance_type == 'tied':
            self.covariances = np.eye(n_features)
        elif self.covariance_type == 'diag':
            self.covariances = np.stack([np.ones(n_features) for _ in range(self.n_components)])
        elif self.covariance_type == 'spherical':
            self.covariances = np.ones(self.n_components)

    def log_gaussian_pdf(self, X):
        """
        Compute the log Gaussian probability density function.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns:
        --------
        log_prob : array, shape (n_samples, n_components)
            Log probabilities of each data point in X.
        """
        n_samples, n_features = X.shape
        log_prob = np.zeros((n_samples, self.n_components))

        if self.covariance_type == 'full':
            for k in range(self.n_components):
                diff = X - self.means[k]
                log_det = np.linalg.slogdet(self.covariances[k])[1]
                log_prob[:, k] = -0.5 * (n_features * np.log(2 * np.pi) + log_det + 
                                         np.sum(diff @ np.linalg.inv(self.covariances[k]) * diff, axis=1))
        elif self.covariance_type == 'tied':
            log_det = np.linalg.slogdet(self.covariances)[1]
            for k in range(self.n_components):
                diff = X - self.means[k]
                log_prob[:, k] = -0.5 * (n_features * np.log(2 * np.pi) + log_det + 
                                         np.sum(diff @ np.linalg.inv(self.covariances) * diff, axis=1))
        elif self.covariance_type == 'diag':
            log_det = np.sum(np.log(self.covariances), axis=1)
            for k in range(self.n_components):
                diff = X - self.means[k]
                log_prob[:, k] = -0.5 * (n_features * np.log(2 * np.pi) + log_det[k] + 
                                         np.sum((diff ** 2) / self.covariances[k], axis=1))
        elif self.covariance_type == 'spherical':
            for k in range(self.n_components):
                diff = X - self.means[k]
                log_prob[:, k] = -0.5 * (n_features * np.log(2 * np.pi) + n_features * np.log(self.covariances[k]) + 
                                         np.sum(diff ** 2, axis=1) / self.covariances[k])

        return log_prob

    def expectation_step(self, X):
        """
        E-step: compute the expectation of the latent variables.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns:
        --------
        resp : array, shape (n_samples, n_components)
            Posterior probabilities of each mixture component for each observation.
        """
        log_prob = self.log_gaussian_pdf(X)
        log_weighted_prob = log_prob + np.log(self.weights)
        log_prob_norm = logsumexp(log_weighted_prob, axis=1)
        log_resp = log_weighted_prob - log_prob_norm[:, np.newaxis]
        return np.exp(log_resp)

    def maximization_step(self, X, resp):
        """
        M-step: maximize the expected complete-data log-likelihood.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data.
        resp : array-like, shape (n_samples, n_components)
            Posterior probabilities of each mixture component for each observation.
        """
        n_samples, n_features = X.shape
        weights = resp.sum(axis=0)
        self.weights = weights / n_samples
        self.means = np.dot(resp.T, X) / weights[:, np.newaxis]

        if self.covariance_type == 'full':
            for k in range(self.n_components):
                diff = X - self.means[k]
                self.covariances[k] = np.dot(resp[:, k] * diff.T, diff) / weights[k] + np.eye(n_features) * self.reg_covar
        elif self.covariance_type == 'tied':
            self.covariances = np.zeros((n_features, n_features))
            for k in range(self.n_components):
                diff = X - self.means[k]
                self.covariances += np.dot(resp[:, k] * diff.T, diff)
            self.covariances /= n_samples
            self.covariances += np.eye(n_features) * self.reg_covar
        elif self.covariance_type == 'diag':
            avg_X2 = np.dot(resp.T, X * X) / weights[:, np.newaxis]
            avg_means2 = self.means ** 2
            avg_X_means = self.means * np.dot(resp.T, X) / weights[:, np.newaxis]
            self.covariances = avg_X2 - 2 * avg_X_means + avg_means2 + self.reg_covar
        elif self.covariance_type == 'spherical':
            avg_X2 = np.dot(resp.T, np.sum(X * X, axis=1)) / weights
            avg_means2 = np.sum(self.means ** 2, axis=1)
            avg_X_means = np.sum(self.means * np.dot(resp.T, X), axis=1) / weights
            self.covariances = avg_X2 - 2 * avg_X_means + avg_means2
            self.covariances /= n_features
            self.covariances += self.reg_covar

    def fit(self, X):
        """
        Estimate model parameters with the EM algorithm.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns:
        --------
        self : object
            Returns self.
        """
        self.initialize_parameters(X)
        prev_log_likelihood = -np.inf

        for _ in range(self.n_iter):
            resp = self.expectation_step(X)
            self.maximization_step(X, resp)
            log_likelihood = self.compute_log_likelihood(X)
            if np.abs(log_likelihood - prev_log_likelihood) < self.tol:
                break
            prev_log_likelihood = log_likelihood

        return self

    def predict(self, X):
        """
        Predict the labels for the data samples in X using trained model.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns:
        --------
        labels : array, shape (n_samples,)
            Component labels.
        """
        resp = self.expectation_step(X)
        return np.argmax(resp, axis=1)

    def compute_log_likelihood(self, X):
        """
        Compute the log-likelihood of the data under the model.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns:
        --------
        log_likelihood : float
            Log-likelihood of the data.
        """
        log_prob = self.log_gaussian_pdf(X)
        return np.sum(logsumexp(np.log(self.weights) + log_prob, axis=1))

    def score_samples(self, X):
        """
        Compute the log-likelihood of each sample.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns:
        --------
        log_prob : array, shape (n_samples,)
            Log-likelihood of each sample.
        """
        log_prob = self.log_gaussian_pdf(X)
        return logsumexp(np.log(self.weights) + log_prob, axis=1)

    def bic(self, X):
        """
        Bayesian Information Criterion for the current model on the input X.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns:
        --------
        bic : float
            BIC for the current model on the input X.
        """
        n_samples, n_features = X.shape
        cov_params = {'full': self.n_components * n_features * (n_features + 1) / 2,
                      'tied': n_features * (n_features + 1) / 2,
                      'diag': self.n_components * n_features,
                      'spherical': self.n_components}
        n_parameters = (self.n_components - 1) + self.n_components * n_features + cov_params[self.covariance_type]
        return -2 * self.compute_log_likelihood(X) + n_parameters * np.log(n_samples)

    def aic(self, X):
        """
        Akaike Information Criterion for the current model on the input X.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns:
        --------
        aic : float
            AIC for the current model on the input X.
        """
        n_samples, n_features = X.shape
        cov_params = {'full': self.n_components * n_features * (n_features + 1) / 2,
                      'tied': n_features * (n_features + 1) / 2,
                      'diag': self.n_components * n_features,
                      'spherical': self.n_components}
        n_parameters = (self.n_components - 1) + self.n_components * n_features + cov_params[self.covariance_type]
        return -2 * self.compute_log_likelihood(X) + 2 * n_parameters