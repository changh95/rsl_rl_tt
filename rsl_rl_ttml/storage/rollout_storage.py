# SPDX-License-Identifier: BSD-3-Clause

"""Rollout storage using numpy arrays.

All data is stored on CPU as numpy arrays. Conversion to ttml tensors happens
only during the PPO update phase when data needs to go through the NPU.
"""

from __future__ import annotations

import numpy as np
from collections.abc import Generator


class RolloutStorage:
    """Experience buffer for on-policy RL, stored as numpy arrays."""

    class Transition:
        """Container for a single step's data."""

        def __init__(self) -> None:
            self.observations: dict[str, np.ndarray] | None = None
            self.actions: np.ndarray | None = None
            self.rewards: np.ndarray | None = None
            self.dones: np.ndarray | None = None
            self.values: np.ndarray | None = None
            self.actions_log_prob: np.ndarray | None = None
            self.action_mean: np.ndarray | None = None
            self.action_std: np.ndarray | None = None

        def clear(self) -> None:
            self.__init__()

    class Batch:
        """A mini-batch of data for PPO updates."""

        def __init__(
            self,
            observations: dict[str, np.ndarray] | None = None,
            actions: np.ndarray | None = None,
            values: np.ndarray | None = None,
            advantages: np.ndarray | None = None,
            returns: np.ndarray | None = None,
            old_actions_log_prob: np.ndarray | None = None,
            old_action_mean: np.ndarray | None = None,
            old_action_std: np.ndarray | None = None,
        ) -> None:
            self.observations = observations
            self.actions = actions
            self.values = values
            self.advantages = advantages
            self.returns = returns
            self.old_actions_log_prob = old_actions_log_prob
            self.old_action_mean = old_action_mean
            self.old_action_std = old_action_std

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        obs_shapes: dict[str, tuple[int, ...]],
        num_actions: int,
    ) -> None:
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        self.num_actions = num_actions

        # Allocate buffers
        self.observations: dict[str, np.ndarray] = {
            key: np.zeros((num_transitions_per_env, num_envs, *shape), dtype=np.float32)
            for key, shape in obs_shapes.items()
        }
        self.actions = np.zeros((num_transitions_per_env, num_envs, num_actions), dtype=np.float32)
        self.rewards = np.zeros((num_transitions_per_env, num_envs, 1), dtype=np.float32)
        self.dones = np.zeros((num_transitions_per_env, num_envs, 1), dtype=np.float32)
        self.values = np.zeros((num_transitions_per_env, num_envs, 1), dtype=np.float32)
        self.actions_log_prob = np.zeros((num_transitions_per_env, num_envs, 1), dtype=np.float32)
        self.returns = np.zeros((num_transitions_per_env, num_envs, 1), dtype=np.float32)
        self.advantages = np.zeros((num_transitions_per_env, num_envs, 1), dtype=np.float32)

        # Distribution parameters for KL computation
        self.action_means = np.zeros((num_transitions_per_env, num_envs, num_actions), dtype=np.float32)
        self.action_stds = np.zeros((num_transitions_per_env, num_envs, num_actions), dtype=np.float32)

        self.step = 0

    def add_transition(self, transition: Transition) -> None:
        """Store one transition at the current step index."""
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow!")

        for key in self.observations:
            self.observations[key][self.step] = transition.observations[key]
        self.actions[self.step] = transition.actions
        self.rewards[self.step] = transition.rewards.reshape(-1, 1)
        self.dones[self.step] = transition.dones.reshape(-1, 1)
        self.values[self.step] = transition.values.reshape(-1, 1)
        self.actions_log_prob[self.step] = transition.actions_log_prob.reshape(-1, 1)
        if transition.action_mean is not None:
            self.action_means[self.step] = transition.action_mean
        if transition.action_std is not None:
            self.action_stds[self.step] = transition.action_std

        self.step += 1

    def clear(self) -> None:
        """Reset the write cursor."""
        self.step = 0

    def mini_batch_generator(
        self, num_mini_batches: int, num_epochs: int = 8
    ) -> Generator[Batch, None, None]:
        """Yield shuffled flat mini-batches for feedforward PPO updates."""
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        # Flatten [T, E, ...] -> [T*E, ...]
        flat_obs = {key: val.reshape(batch_size, -1) for key, val in self.observations.items()}
        flat_actions = self.actions.reshape(batch_size, -1)
        flat_values = self.values.reshape(batch_size, -1)
        flat_returns = self.returns.reshape(batch_size, -1)
        flat_old_log_prob = self.actions_log_prob.reshape(batch_size, -1)
        flat_advantages = self.advantages.reshape(batch_size, -1)
        flat_old_mean = self.action_means.reshape(batch_size, -1)
        flat_old_std = self.action_stds.reshape(batch_size, -1)

        for _epoch in range(num_epochs):
            indices = np.random.permutation(num_mini_batches * mini_batch_size)
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size
                batch_idx = indices[start:stop]

                yield RolloutStorage.Batch(
                    observations={key: val[batch_idx] for key, val in flat_obs.items()},
                    actions=flat_actions[batch_idx],
                    values=flat_values[batch_idx],
                    advantages=flat_advantages[batch_idx],
                    returns=flat_returns[batch_idx],
                    old_actions_log_prob=flat_old_log_prob[batch_idx],
                    old_action_mean=flat_old_mean[batch_idx],
                    old_action_std=flat_old_std[batch_idx],
                )
