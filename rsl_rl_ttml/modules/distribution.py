# SPDX-License-Identifier: BSD-3-Clause

"""Gaussian action distribution using numpy for sampling and log-prob computation.

The log_std is a learnable parameter updated via a simple gradient step
during PPO updates, alongside the MLP parameters on the NPU.
"""

from __future__ import annotations

import numpy as np


class GaussianDistribution:
    """Gaussian distribution with learnable state-independent std."""

    LOG_2PI = np.log(2.0 * np.pi)

    def __init__(self, output_dim: int, init_std: float = 1.0) -> None:
        self.output_dim = output_dim
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
        return mlp_output

    def log_prob(self, actions: np.ndarray) -> np.ndarray:
        """Compute log probability, summed over action dim. Shape: [B]."""
        std = self.std
        var = std ** 2
        log_p = -0.5 * ((actions - self._mean) ** 2 / var + self.LOG_2PI + 2.0 * self.log_std)
        return log_p.sum(axis=-1)

    def entropy(self) -> np.ndarray:
        """Compute entropy, summed over action dim. Shape: [B]."""
        batch_size = self._mean.shape[0]
        ent = 0.5 * self.output_dim * (1.0 + self.LOG_2PI) + self.log_std.sum()
        return np.full(batch_size, ent, dtype=np.float32)

    @property
    def params(self) -> tuple[np.ndarray, np.ndarray]:
        return (self._mean.copy(), self.std.copy())

    @staticmethod
    def kl_divergence(
        old_params: tuple[np.ndarray, np.ndarray],
        new_params: tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        old_mean, old_std = old_params
        new_mean, new_std = new_params
        log_ratio = np.log(new_std + 1e-8) - np.log(old_std + 1e-8)
        kl = log_ratio + (old_std ** 2 + (old_mean - new_mean) ** 2) / (2.0 * new_std ** 2 + 1e-8) - 0.5
        return kl.sum(axis=-1)

    def update_std(self, actions: np.ndarray, advantages: np.ndarray, lr: float = 1e-3) -> None:
        """Update log_std using the REINFORCE gradient.

        The gradient of the surrogate loss w.r.t. log_std is:
        d(loss)/d(log_std) = -advantage * d(log_prob)/d(log_std)
        where d(log_prob)/d(log_std) = ((action - mean)^2 / var - 1)

        Args:
            actions: Sampled actions [B, A]
            advantages: Per-sample advantages [B]
            lr: Learning rate for std update
        """
        std = self.std
        var = std ** 2
        # d(log_prob)/d(log_std) = ((action - mean)^2 / var - 1)
        d_logprob_d_logstd = (actions - self._mean) ** 2 / var - 1.0  # [B, A]
        # Policy gradient for log_std
        grad = -(advantages[:, None] * d_logprob_d_logstd).mean(axis=0)  # [A]
        # Gradient step
        self.log_std -= lr * grad
        # Clamp log_std to reasonable range
        self.log_std = np.clip(self.log_std, -3.0, 2.0)  # std in [0.05, 7.4]
