import numpy as np
import scipy.io as scio
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import time
import h5py


class FuzzySVM:
    def __init__(self, lambda_param=0.01, tau=0.1, s_plus=0.7, s_minus=0.5, max_iter=100, tol=1e-4):
        self.lambda_param = lambda_param
        self.tau = tau
        self.s_plus = s_plus
        self.s_minus = s_minus
        self.max_iter = max_iter
        self.tol = tol
        self.theta = None
        self.theta0 = None

    def phi(self, u):
        if u <= -1:
            return 0
        elif -1 < u < 1:
            return 0.5 + (15 / 16) * (u - (2 / 3) * u ** 3 + (1 / 5) * u ** 5)
        else:
            return 1

    def phi_derivative(self, u):
        if u <= -1 or u >= 1:
            return 0
        else:
            return (15 / 16) * (1 - 2 * u ** 2 + u ** 4)

    def compute_membership(self, X, y):
        s = np.zeros(len(y))
        s[y == 1] = self.s_plus
        s[y == -1] = self.s_minus
        return s

    def fit(self, X, y, distributed=False, N=5, T=5):
        n, p = X.shape
        X_aug = np.column_stack([np.ones(n), X])
        s = self.compute_membership(X, y)

        if distributed:
            partitions = np.array_split(np.arange(n), N)
            theta = self._initial_estimator(X_aug[partitions[0]], y[partitions[0]], s[partitions[0]])

            for _ in range(self.max_iter):
                A_total = np.zeros(p + 1)
                B_total = np.zeros((p + 1, p + 1))

                for k in range(N):
                    idx = partitions[k]
                    X_k = X_aug[idx]
                    y_k = y[idx]
                    s_k = s[idx]

                    v_k = 1 - y_k * np.dot(X_k, theta)
                    u_k = v_k / self.tau

                    phi_k = np.array([self.phi(u) for u in u_k])
                    phi_deriv_k = np.array([self.phi_derivative(u) for u in u_k])

                    A_k = (1 / n) * np.sum(s_k[:, np.newaxis] * y_k[:, np.newaxis] * X_k *
                                           (phi_k[:, np.newaxis] + (1 / self.tau) * phi_deriv_k[:, np.newaxis]), axis=0)

                    outer_prods = np.array([np.outer(xi, xi) for xi in X_k])
                    weighted_outer = (s_k[:, np.newaxis, np.newaxis] / self.tau) * outer_prods * phi_deriv_k[:,
                                                                                                 np.newaxis, np.newaxis]
                    B_k = (1 / n) * np.sum(weighted_outer, axis=0)

                    A_total += A_k
                    B_total += B_k

                B1 = B_total / N
                for _ in range(T):
                    theta_new = theta - np.linalg.inv(N * B1 + 1e-6 * np.eye(p + 1)) @ (
                            B_total @ theta - A_total + self.lambda_param * np.concatenate([[0], theta[1:]]))
                    theta = theta_new

                if np.linalg.norm(theta_new - theta) < self.tol:
                    break

            self.theta = theta[1:]
            self.theta0 = theta[0]
        else:
            theta = self._initial_estimator(X_aug, y, s)
            self.theta = theta[1:]
            self.theta0 = theta[0]

    def _initial_estimator(self, X_aug, y, s):
        n, p = X_aug.shape
        theta = np.random.randn(p)

        for _ in range(self.max_iter):
            margin = y * np.dot(X_aug, theta)
            hinge_grad = np.where(margin < 1, -1, 0)

            grad = (1 / n) * np.sum((s[:, np.newaxis] * hinge_grad[:, np.newaxis] *
                                     y[:, np.newaxis] * X_aug), axis=0) + self.lambda_param * np.concatenate(
                [[0], theta[1:]])

            theta_new = theta - 0.01 * grad

            if np.linalg.norm(theta_new - theta) < self.tol:
                break

            theta = theta_new

        return theta

    def predict(self, X):
        if self.theta is None or self.theta0 is None:
            raise ValueError("Model not trained yet.")
        scores = self.theta0 + np.dot(X, self.theta)
        return np.where(scores >= 0, 1, -1)
