# SPDX-License-Identifier: BSD-3-Clause

"""Abstract vectorized environment interface for ttml-based RL.

Environments produce numpy arrays on CPU. The policy handles conversion to/from
ttml tensors on the Tenstorrent NPU.
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod


class VecEnv(ABC):
    """Abstract vectorized environment using numpy arrays."""

    num_envs: int
    num_actions: int
    max_episode_length: int
    episode_length_buf: np.ndarray

    @abstractmethod
    def get_observations(self) -> dict[str, np.ndarray]:
        """Return current observations as {group_name: array[num_envs, obs_dim]}."""
        raise NotImplementedError

    @abstractmethod
    def step(self, actions: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict]:
        """Apply actions and return (obs, rewards, dones, extras).

        Args:
            actions: Shape [num_envs, num_actions].

        Returns:
            obs: Dict of observation arrays.
            rewards: Shape [num_envs].
            dones: Shape [num_envs].
            extras: Dict with optional "time_outs" array and "log" dict.
        """
        raise NotImplementedError
