class RSLDA:
    def __init__(self, n_components=1, max_iter=100, tol=1e-6):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.W = None
        self.lambdas = None

    def fit(self, X, y):
        Sb, Sw = self._calculate_scatter_matrices(X, y)
        d = X.shape[1]

        self.W = np.random.rand(d, self.n_components)
        self.W, _ = np.linalg.qr(self.W)

        prev_obj = -np.inf
        for _ in range(self.max_iter):
            self.lambdas = np.zeros(self.n_components)
            for k in range(self.n_components):
                wk = self.W[:, k]
                numerator = np.sqrt(wk.T @ Sb @ wk)
                denominator = wk.T @ Sw @ wk
                self.lambdas[k] = numerator / denominator

            gamma = np.trace(Sw) * 10
            S = gamma * np.eye(d) - Sw
            C = np.zeros((d, self.n_components))
            for k in range(self.n_components):
                Ak = self.lambdas[k] ** 2 * S
                Bk = (self.lambdas[k] * Sb) / np.sqrt(self.W[:, k].T @ Sb @ self.W[:, k])
                ck = 2 * (Ak + Bk) @ self.W[:, k]
                C[:, k] = ck

            U, _, Vh = np.linalg.svd(C, full_matrices=False)
            self.W = U[:, :self.n_components] @ Vh

            current_obj = np.sum([(self.lambdas[k] ** 2 * self.W[:, k].T @ Sw @ self.W[:, k] -
                                   2 * self.lambdas[k] * np.sqrt(self.W[:, k].T @ Sb @ self.W[:, k]))
                                  for k in range(self.n_components)])
            if np.abs(current_obj - prev_obj) < self.tol:
                break
            prev_obj = current_obj

