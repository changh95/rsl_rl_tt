# SPDX-License-Identifier: BSD-3-Clause

"""Empirical normalization using numpy running statistics.

Normalization stats are maintained on CPU (numpy). The normalize/denormalize
operations are applied before/after ttml tensor conversion.
"""

from __future__ import annotations

import numpy as np


class EmpiricalNormalization:
    """Normalize mean and variance of values based on empirical values (CPU/numpy)."""

    def __init__(self, shape: int, eps: float = 1e-2, until: int | None = None) -> None:
        self.eps = eps
        self.until = until
        self._mean = np.zeros(shape, dtype=np.float32)
        self._var = np.ones(shape, dtype=np.float32)
        self._std = np.ones(shape, dtype=np.float32)
        self.count = 0
        self._training = True

    def train(self):
        self._training = True

    def eval(self):
        self._training = False

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize x using running stats. x shape: [B, D]."""
        return (x - self._mean) / (self._std + self.eps)

    def inverse(self, y: np.ndarray) -> np.ndarray:
        """De-normalize values."""
        return y * (self._std + self.eps) + self._mean

    def update(self, x: np.ndarray) -> None:
        """Update running mean/var from batch x of shape [B, D]."""
        if not self._training:
            return
        if self.until is not None and self.count >= self.until:
            return

        count_x = x.shape[0]
        self.count += count_x
        rate = count_x / self.count
        var_x = np.var(x, axis=0)
        mean_x = np.mean(x, axis=0)
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))
        self._std = np.sqrt(self._var)

    def state_dict(self) -> dict:
        return {
            "mean": self._mean.copy(),
            "var": self._var.copy(),
            "std": self._std.copy(),
            "count": self.count,
        }

    def load_state_dict(self, state: dict) -> None:
        self._mean = state["mean"].copy()
        self._var = state["var"].copy()
        self._std = state["std"].copy()
        self.count = state["count"]
