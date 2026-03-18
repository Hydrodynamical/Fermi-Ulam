"""Lightweight hazard models with a sklearn-like predict_proba interface.

The current workspace uses a Python 3.15 alpha interpreter, where prebuilt
scikit-learn wheels are not available. These models keep the hazard-learning
pipeline runnable by providing a small, serializable classifier API that can be
saved with joblib and later consumed by evaluation scripts.
"""

from __future__ import annotations

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


class PolynomialLogisticHazardModel:
    """Polynomial-basis logistic hazard model trained with mini-batch Adam.

    Parameters
    ----------
    degree : int
        Maximum polynomial degree in the standardized input features.
    learning_rate : float
        Adam step size.
    max_epochs : int
        Maximum number of epochs.
    batch_size : int
        Mini-batch size.
    l2 : float
        L2 regularization strength.
    random_state : int
        RNG seed.
    validation_fraction : float
        Fraction of training rows reserved for early stopping.
    n_iter_no_change : int
        Early-stopping patience in epochs.
    verbose : bool
        Whether to print epoch-level progress during fit.
    """

    def __init__(
        self,
        degree: int = 3,
        learning_rate: float = 0.03,
        max_epochs: int = 60,
        batch_size: int = 8192,
        l2: float = 1e-4,
        random_state: int = 0,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 8,
        verbose: bool = False,
    ) -> None:
        self.degree = degree
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.random_state = random_state
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.verbose = verbose

    def _standardize_fit(self, X: np.ndarray) -> np.ndarray:
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_ = np.where(self.scale_ > 0, self.scale_, 1.0)
        return (X - self.mean_) / self.scale_

    def _standardize_transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_

    def _feature_map(self, X: np.ndarray) -> np.ndarray:
        cols = [np.ones((X.shape[0], 1), dtype=np.float64)]
        cols.append(X)

        if self.degree >= 2:
            quad = []
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    quad.append((X[:, i] * X[:, j]).reshape(-1, 1))
            cols.extend(quad)

        if self.degree >= 3:
            cubic = []
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    for k in range(j, X.shape[1]):
                        cubic.append((X[:, i] * X[:, j] * X[:, k]).reshape(-1, 1))
            cols.extend(cubic)

        return np.hstack(cols)

    def _log_loss(self, X_phi: np.ndarray, y: np.ndarray) -> float:
        p = _sigmoid(X_phi @ self.coef_)
        eps = 1e-8
        loss = -np.mean(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps))
        return float(loss + 0.5 * self.l2 * np.sum(self.coef_[1:] ** 2))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PolynomialLogisticHazardModel":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")

        rng = np.random.default_rng(self.random_state)
        perm = rng.permutation(len(y))
        X = X[perm]
        y = y[perm]

        n_val = int(round(self.validation_fraction * len(y)))
        n_val = min(max(n_val, 1), max(len(y) - 1, 1)) if len(y) > 1 else 0

        if n_val > 0:
            X_train, X_val = X[:-n_val], X[-n_val:]
            y_train, y_val = y[:-n_val], y[-n_val:]
        else:
            X_train, X_val = X, X[:0]
            y_train, y_val = y, y[:0]

        X_train_std = self._standardize_fit(X_train)
        X_train_phi = self._feature_map(X_train_std)
        X_val_phi = self._feature_map(self._standardize_transform(X_val)) if len(X_val) else None

        self.coef_ = np.zeros(X_train_phi.shape[1], dtype=np.float64)
        m = np.zeros_like(self.coef_)
        v = np.zeros_like(self.coef_)
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        best_coef = self.coef_.copy()
        best_val = np.inf
        patience = 0
        step = 0
        self.n_iter_ = 0

        for epoch in range(self.max_epochs):
            epoch_perm = rng.permutation(len(y_train))
            X_epoch = X_train_phi[epoch_perm]
            y_epoch = y_train[epoch_perm]

            for start in range(0, len(y_epoch), self.batch_size):
                stop = min(start + self.batch_size, len(y_epoch))
                X_batch = X_epoch[start:stop]
                y_batch = y_epoch[start:stop]

                p = _sigmoid(X_batch @ self.coef_)
                grad = (X_batch.T @ (p - y_batch)) / len(y_batch)
                grad[1:] += self.l2 * self.coef_[1:]

                step += 1
                m = beta1 * m + (1.0 - beta1) * grad
                v = beta2 * v + (1.0 - beta2) * (grad * grad)
                m_hat = m / (1.0 - beta1 ** step)
                v_hat = v / (1.0 - beta2 ** step)
                self.coef_ -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

            self.n_iter_ = epoch + 1

            if X_val_phi is None or len(X_val_phi) == 0:
                continue

            val_loss = self._log_loss(X_val_phi, y_val)
            if self.verbose:
                print(f"    epoch={epoch + 1:03d}  val_log_loss={val_loss:.5f}")

            if val_loss + 1e-5 < best_val:
                best_val = val_loss
                best_coef = self.coef_.copy()
                patience = 0
            else:
                patience += 1
                if patience >= self.n_iter_no_change:
                    break

        self.coef_ = best_coef
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        X_phi = self._feature_map(self._standardize_transform(X))
        p = _sigmoid(X_phi @ self.coef_)
        return np.column_stack([1.0 - p, p])