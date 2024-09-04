import numpy as np
from scipy.special import logsumexp

class GMM:
    def __init__(self, n_components, covariance_type='full', n_iter=100, tol=1e-5, reg_covar=1e-6, random_state=None):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state

    def initialize_parameters(self, X):
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
        log_prob = self.log_gaussian_pdf(X)
        log_weighted_prob = log_prob + np.log(self.weights)
        log_prob_norm = logsumexp(log_weighted_prob, axis=1)
        log_resp = log_weighted_prob - log_prob_norm[:, np.newaxis]
        return np.exp(log_resp)

    def maximization_step(self, X, resp):
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
        resp = self.expectation_step(X)
        return np.argmax(resp, axis=1)

    def compute_log_likelihood(self, X):
        log_prob = self.log_gaussian_pdf(X)
        return np.sum(logsumexp(np.log(self.weights) + log_prob, axis=1))

    def score_samples(self, X):
        log_prob = self.log_gaussian_pdf(X)
        return logsumexp(np.log(self.weights) + log_prob, axis=1)

    def bic(self, X):
        """Bayesian Information Criterion"""
        n_samples, n_features = X.shape
        cov_params = {'full': self.n_components * n_features * (n_features + 1) / 2,
                      'tied': n_features * (n_features + 1) / 2,
                      'diag': self.n_components * n_features,
                      'spherical': self.n_components}
        n_parameters = (self.n_components - 1) + self.n_components * n_features + cov_params[self.covariance_type]
        return -2 * self.compute_log_likelihood(X) + n_parameters * np.log(n_samples)

    def aic(self, X):
        """Akaike Information Criterion"""
        n_samples, n_features = X.shape
        cov_params = {'full': self.n_components * n_features * (n_features + 1) / 2,
                      'tied': n_features * (n_features + 1) / 2,
                      'diag': self.n_components * n_features,
                      'spherical': self.n_components}
        n_parameters = (self.n_components - 1) + self.n_components * n_features + cov_params[self.covariance_type]
        return -2 * self.compute_log_likelihood(X) + 2 * n_parameters