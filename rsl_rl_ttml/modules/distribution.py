# SPDX-License-Identifier: BSD-3-Clause

"""Gaussian action distribution using numpy for sampling and log-prob computation.

Sampling uses the reparameterization trick on CPU. Log probabilities and entropy
are computed analytically using numpy for the rollout phase (no grad needed).

During the PPO update, log_prob and entropy are recomputed on the NPU using ttml ops.
"""

from __future__ import annotations

import numpy as np


class GaussianDistribution:
    """Gaussian distribution with state-independent std (numpy-based).

    The std is a learnable parameter stored as numpy and synced with the ttml
    parameter during optimization.
    """

    LOG_2PI = np.log(2.0 * np.pi)

    def __init__(self, output_dim: int, init_std: float = 1.0) -> None:
        self.output_dim = output_dim
        # Learnable std parameter (will be exposed to optimizer)
        self.log_std = np.log(np.full(output_dim, init_std, dtype=np.float32))
        self._mean: np.ndarray | None = None

    @property
    def std(self) -> np.ndarray:
        return np.exp(self.log_std)

    @property
    def mean(self) -> np.ndarray:
        return self._mean

    def update(self, mlp_output: np.ndarray) -> None:
        """Set the mean from MLP output [B, D]."""
        self._mean = mlp_output

    def sample(self) -> np.ndarray:
        """Sample actions using the reparameterization trick."""
        noise = np.random.randn(*self._mean.shape).astype(np.float32)
        return self._mean + self.std * noise

    def deterministic_output(self, mlp_output: np.ndarray) -> np.ndarray:
        """Return the mean (deterministic action)."""
        return mlp_output

    def log_prob(self, actions: np.ndarray) -> np.ndarray:
        """Compute log probability, summed over action dim. Shape: [B]."""
        std = self.std
        var = std ** 2
        log_std = self.log_std
        # log p(x) = -0.5 * ((x - mu)^2 / var + log(2*pi) + 2*log_std)
        log_p = -0.5 * ((actions - self._mean) ** 2 / var + self.LOG_2PI + 2.0 * log_std)
        return log_p.sum(axis=-1)  # [B]

    def entropy(self) -> np.ndarray:
        """Compute entropy, summed over action dim. Shape: [B]."""
        # H = 0.5 * D * (1 + log(2*pi)) + sum(log_std)
        batch_size = self._mean.shape[0]
        ent = 0.5 * self.output_dim * (1.0 + self.LOG_2PI) + self.log_std.sum()
        return np.full(batch_size, ent, dtype=np.float32)

    @property
    def params(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, std) for KL computation."""
        return (self._mean.copy(), self.std.copy())

    @staticmethod
    def kl_divergence(
        old_params: tuple[np.ndarray, np.ndarray],
        new_params: tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        """Compute KL(old || new) summed over action dim. Shape: [B]."""
        old_mean, old_std = old_params
        new_mean, new_std = new_params
        log_ratio = np.log(new_std) - np.log(old_std)
        kl = log_ratio + (old_std ** 2 + (old_mean - new_mean) ** 2) / (2.0 * new_std ** 2) - 0.5
        return kl.sum(axis=-1)  # [B]
