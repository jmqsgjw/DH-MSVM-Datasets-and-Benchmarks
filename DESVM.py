import numpy as np
from scipy.io import loadmat
from scipy.linalg import svd
from sklearn.metrics import accuracy_score, precision_score, recall_score
import time
import h5py


class DESVM:
    def __init__(self, lambda_param=1.0, gamma=1.0, max_iter=100, tol=1e-4):
        self.lambda_param = lambda_param
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.a = 0.0
        self.A = None
        self.B = None
        self.u = None
        self.V = None
        self.r = None
        self.s = None

    def fit(self, X, y):
        n, p, q = X.shape
        self.a = 0.0
        self.A = np.zeros((p, q))
        self.B = np.zeros((p, q))
        self.u = np.zeros(n)
        self.V = np.zeros((p, q))
        self.r = np.zeros(n)
        self.s = np.zeros(p * q)

        for iteration in range(self.max_iter):
            # Update A
            X_vec = X.reshape(n, -1)
            Y = np.diag(y)
            A_vec = np.linalg.inv(X_vec.T @ X_vec + np.eye(p * q)) @ (
                    X_vec.T @ Y @ (self.u / self.gamma + np.ones(n) - self.r) -
                    X_vec.T @ np.ones(n) * self.a + self.s / self.gamma +
                    self.B.reshape(-1) - self.V.reshape(-1) / self.gamma
            )
            self.A = A_vec.reshape(p, q)

            # Update B using SVD
            U, s_vals, Vt = svd(self.A + self.V / self.gamma, full_matrices=False)
            s_thresh = np.maximum(s_vals - self.lambda_param / self.gamma, 0)
            self.B = U @ np.diag(s_thresh) @ Vt

            # Update a
            self.a = (np.sum(y * (self.u / self.gamma + np.ones(n) - self.r)) / (n * self.gamma) +
                      np.mean(y * (X_vec @ self.A.reshape(-1) + self.a)))

            # Update r
            e = np.ones(n) - y * (X_vec @ self.A.reshape(-1) + self.a)
            self.r = self.soft_threshold(self.u / self.gamma + e, 1 / (n * self.gamma))

            # Update dual variables
            self.u += self.gamma * (e - self.r)
            self.V += self.gamma * (self.A - self.B)

            # Check convergence
            if np.linalg.norm(self.A - self.B) < self.tol:
                break

    def predict(self, X):
        n, p, q = X.shape
        X_vec = X.reshape(n, -1)
        scores = X_vec @ self.A.reshape(-1) + self.a
        return np.sign(scores)

    @staticmethod
    def soft_threshold(x, threshold):
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


def find_optimal_factors(n_features):
    factors = []
    for i in range(1, int(np.sqrt(n_features)) + 1):
        if n_features % i == 0:
            factors.append((i, n_features // i))
    if not factors:
        return 1, n_features
    # 选择最接近的因数对
    return factors[-1]

