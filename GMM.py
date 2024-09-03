import numpy as np

class GMM:
    def __init__(self, n_components, n_iter=100, tol=1e-5):
        self.n_components = n_components
        self.n_iter = n_iter
        self.tol = tol

    def initialize_parameters(self, X):
        n_samples, n_features = X.shape
        self.weights = np.full(self.n_components, 1/self.n_components)
        self.means = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.covariances = [np.eye(n_features) for _ in range(self.n_components)]

    def gaussian_pdf(self, X, mean, cov):
        n_dim = len(mean)
        diff = X - mean
        return (1. / (np.sqrt((2 * np.pi)**n_dim * np.linalg.det(cov))) * 
                np.exp(-0.5 * np.sum(diff @ np.linalg.inv(cov) * diff, axis=1)))

    def expectation_step(self, X):
        responsibilities = np.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            responsibilities[:, k] = self.weights[k] * self.gaussian_pdf(X, self.means[k], self.covariances[k])
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        return responsibilities

    def maximization_step(self, X, responsibilities):
        total_responsibility = responsibilities.sum(axis=0)
        self.weights = total_responsibility / X.shape[0]
        self.means = responsibilities.T @ X / total_responsibility[:, np.newaxis]
        for k in range(self.n_components):
            diff = X - self.means[k]
            self.covariances[k] = (responsibilities[:, k, np.newaxis] * diff).T @ diff / total_responsibility[k]

    def compute_log_likelihood(self, X):
        likelihood = np.zeros(X.shape[0])
        for k in range(self.n_components):
            likelihood += self.weights[k] * self.gaussian_pdf(X, self.means[k], self.covariances[k])
        return np.sum(np.log(likelihood))

    def fit(self, X):
        self.initialize_parameters(X)
        prev_log_likelihood = -np.inf
        for _ in range(self.n_iter):
            responsibilities = self.expectation_step(X)
            self.maximization_step(X, responsibilities)
            log_likelihood = self.compute_log_likelihood(X)
            if np.abs(log_likelihood - prev_log_likelihood) < self.tol:
                break
            prev_log_likelihood = log_likelihood
        return log_likelihood

    def predict(self, X):
        responsibilities = self.expectation_step(X)
        return np.argmax(responsibilities, axis=1)