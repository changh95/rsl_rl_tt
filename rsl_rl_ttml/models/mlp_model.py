# SPDX-License-Identifier: BSD-3-Clause

"""MLP actor-critic model for ttml.

Handles conversion between numpy [B, D] and ttml [1, 1, B_pad, D_pad],
including tile-alignment padding.
"""

from __future__ import annotations

import numpy as np

from rsl_rl_ttml.modules.mlp import MLP
from rsl_rl_ttml.modules.normalization import EmpiricalNormalization
from rsl_rl_ttml.modules.distribution import GaussianDistribution
from rsl_rl_ttml.utils.tensor_utils import numpy_to_ttml, ttml_to_numpy


class MLPModel:
    """MLP-based actor or critic model.

    For the actor: outputs actions via a Gaussian distribution.
    For the critic: outputs a scalar value estimate.

    During rollouts (act/value), operates in numpy for efficiency.
    During training updates, runs forward passes on NPU via ttml.
    """

    def __init__(
        self,
        obs_groups: list[str],
        obs_dims: dict[str, int],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
    ) -> None:
        self.obs_groups = obs_groups
        self.obs_dim = sum(obs_dims[g] for g in obs_groups)
        self.output_dim = output_dim

        # Observation normalization (CPU/numpy)
        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.obs_dim)
        else:
            self.obs_normalizer = None

        # Distribution (for actor only)
        if distribution_cfg is not None:
            init_std = distribution_cfg.get("init_std", 1.0)
            self.distribution = GaussianDistribution(output_dim, init_std=init_std)
            mlp_output_dim = output_dim
        else:
            self.distribution = None
            mlp_output_dim = output_dim

        # MLP (runs on NPU via ttml)
        self.mlp = MLP(self.obs_dim, mlp_output_dim, hidden_dims, activation)

        self._training = True

    def train(self):
        self._training = True
        if self.obs_normalizer:
            self.obs_normalizer.train()

    def eval(self):
        self._training = False
        if self.obs_normalizer:
            self.obs_normalizer.eval()

    def _get_latent_np(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """Concatenate and normalize observations. Returns [B, D] numpy."""
        parts = [obs[g] for g in self.obs_groups]
        latent = np.concatenate(parts, axis=-1)
        if self.obs_normalizer is not None:
            latent = self.obs_normalizer.normalize(latent)
        return latent.astype(np.float32)

    def forward_np(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """Forward pass returning numpy output. Runs MLP on NPU, converts back."""
        latent = self._get_latent_np(obs)
        B = latent.shape[0]
        # Convert to ttml (handles padding) and run on NPU
        ttml_input = numpy_to_ttml(latent)
        ttml_output = self.mlp(ttml_input)
        # Convert back, unpadding to original dimensions
        return ttml_to_numpy(ttml_output, original_shape=(B, self.output_dim))

    def forward_ttml(self, latent_ttml):
        """Forward pass staying in ttml tensor space (for training updates on NPU)."""
        return self.mlp(latent_ttml)

    def act(self, obs: dict[str, np.ndarray], stochastic: bool = True) -> np.ndarray:
        """Compute actions from observations (numpy in, numpy out)."""
        mlp_out = self.forward_np(obs)
        if self.distribution is not None:
            self.distribution.update(mlp_out)
            if stochastic:
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_out)
        return mlp_out

    def get_value(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """Compute value estimate (for critic). Returns [B, 1]."""
        return self.forward_np(obs)

    def update_normalization(self, obs: dict[str, np.ndarray]) -> None:
        """Update observation normalization statistics."""
        if self.obs_normalizer is not None:
            parts = [obs[g] for g in self.obs_groups]
            latent = np.concatenate(parts, axis=-1)
            self.obs_normalizer.update(latent)

    def parameters(self):
        """Return ttml named parameters for the optimizer."""
        return self.mlp.named_parameters()

    def state_dict(self) -> dict:
        """Return serializable state."""
        import ttnn
        state = {"mlp": {}}
        for name, param in self.mlp.named_parameters():
            state["mlp"][name] = param.tensor.to_numpy(ttnn.DataType.FLOAT32)
        if self.obs_normalizer is not None:
            state["obs_normalizer"] = self.obs_normalizer.state_dict()
        if self.distribution is not None:
            state["distribution_log_std"] = self.distribution.log_std.copy()
        return state

    def load_state_dict(self, state: dict) -> None:
        """Load from serialized state."""
        if self.obs_normalizer is not None and "obs_normalizer" in state:
            self.obs_normalizer.load_state_dict(state["obs_normalizer"])
        if self.distribution is not None and "distribution_log_std" in state:
            self.distribution.log_std = state["distribution_log_std"].copy()
