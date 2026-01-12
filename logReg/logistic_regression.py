import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000, l2_lambda=0.0):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.l2_lambda = l2_lambda
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _loss(self, y, y_hat):
        # Binary cross-entropy loss
        eps = 1e-15  # to avoid log(0)
        y_hat = np.clip(y_hat, eps, 1 - eps)
        base_loss = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        # L2 regularization term scaled by (1 / (2 * n_samples)) to match gradient
        n_samples = y.shape[0]
        l2_penalty = (self.l2_lambda / (2 * n_samples)) * np.sum(self.weights ** 2)
        return base_loss + l2_penalty

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iters):
            linear_output = np.dot(X, self.weights) + self.bias

            # Apply sigmoid
            y_hat = self._sigmoid(linear_output)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_hat - y))
            dw += (self.l2_lambda / n_samples) * self.weights
            db = (1 / n_samples) * np.sum(y_hat - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_output)

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)