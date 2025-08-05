import h5py
import numpy as np
import scipy.io as scio
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time


class dRSVOR:
    def __init__(self, kernel_width=0.8, alpha=0.001, max_iter=1000, tol=1e-4):
        self.kernel_width = kernel_width
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def _gaussian_kernel(self, x):
        return np.exp(-x ** 2 / (2 * self.kernel_width ** 2))

    def _correntropy_loss(self, e):
        return 1 - self._gaussian_kernel(e)

    def _projection(self, b):
        return np.sort(b)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = np.zeros(1)

        for iteration in range(self.max_iter):
            w_prev = np.copy(self.w)
            b_prev = np.copy(self.b)

            # Compute errors
            z = y * (X.dot(self.w) - self.b)
            e = np.maximum(0, 1 - z)

            # Compute weights
            u = (1 - np.exp(-1 / (2 * self.kernel_width ** 2))) * np.exp(-e ** 2 / (2 * self.kernel_width ** 2))

            # Update w
            grad_w = self.alpha * self.w - np.mean(u[:, np.newaxis] * y[:, np.newaxis] * X, axis=0)
            self.w -= grad_w

            # Update b
            grad_b = -np.mean(u * y)
            self.b -= grad_b

            # Project b to feasible region
            self.b = self._projection(self.b)

            # Check for convergence
            if np.linalg.norm(self.w - w_prev) < self.tol and np.linalg.norm(self.b - b_prev) < self.tol:
                break

    def predict(self, X):
        return np.sign(X.dot(self.w) - self.b)
