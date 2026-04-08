#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end test for rsl_rl_ttml: PPO training on a dummy environment.

This validates the full pipeline:
  DummyVecEnv (CPU/numpy) -> rollout collection -> PPO update (NPU via ttml)

Usage:
    python -m rsl_rl_ttml.test_e2e
"""

from __future__ import annotations

import numpy as np
from rsl_rl_ttml.env.vec_env import VecEnv


class DummyVecEnv(VecEnv):
    """Simple continuous-control environment for testing.

    State: 2D position. Action: 2D velocity. Goal: reach origin.
    """

    def __init__(self, num_envs: int = 64) -> None:
        self.num_envs = num_envs
        self.num_actions = 4
        self.max_episode_length = 100
        self.episode_length_buf = np.zeros(num_envs, dtype=np.int32)
        self._obs_dim = 8
        self._state = np.random.randn(num_envs, self._obs_dim).astype(np.float32) * 0.1

    def get_observations(self) -> dict[str, np.ndarray]:
        return {"policy": self._state.copy()}

    def step(self, actions: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict]:
        # Simple dynamics: state += action * 0.1
        self._state[:, :self.num_actions] += actions * 0.1
        self._state += np.random.randn(*self._state.shape).astype(np.float32) * 0.01

        # Reward: negative distance from origin
        rewards = -np.linalg.norm(self._state, axis=-1).astype(np.float32)

        # Episode management
        self.episode_length_buf += 1
        dones = (self.episode_length_buf >= self.max_episode_length).astype(np.float32)
        time_outs = dones.copy()

        # Reset done envs
        done_mask = dones > 0.5
        if done_mask.any():
            self._state[done_mask] = np.random.randn(done_mask.sum(), self._obs_dim).astype(np.float32) * 0.1
            self.episode_length_buf[done_mask] = 0

        obs = {"policy": self._state.copy()}
        extras = {"time_outs": time_outs}
        return obs, rewards, dones, extras


def test_numpy_components():
    """Test all numpy-based components without requiring ttml/NPU."""
    print("=" * 60)
    print("Testing numpy-based components (no NPU required)")
    print("=" * 60)

    # Test environment
    env = DummyVecEnv(num_envs=16)
    obs = env.get_observations()
    assert "policy" in obs
    assert obs["policy"].shape == (16, 8)
    print("[PASS] DummyVecEnv: observations shape correct")

    actions = np.random.randn(16, 4).astype(np.float32)
    obs, rewards, dones, extras = env.step(actions)
    assert rewards.shape == (16,)
    assert dones.shape == (16,)
    print("[PASS] DummyVecEnv: step returns correct shapes")

    # Test normalization
    from rsl_rl_ttml.modules.normalization import EmpiricalNormalization

    norm = EmpiricalNormalization(8)
    data = np.random.randn(100, 8).astype(np.float32)
    norm.update(data)
    normalized = norm.normalize(data)
    assert abs(normalized.mean()) < 0.5  # roughly centered
    print(f"[PASS] EmpiricalNormalization: mean={normalized.mean():.4f}, std={normalized.std():.4f}")

    # Test distribution
    from rsl_rl_ttml.modules.distribution import GaussianDistribution

    dist = GaussianDistribution(output_dim=4, init_std=1.0)
    mean = np.random.randn(16, 4).astype(np.float32)
    dist.update(mean)
    samples = dist.sample()
    assert samples.shape == (16, 4)
    log_prob = dist.log_prob(samples)
    assert log_prob.shape == (16,)
    entropy = dist.entropy()
    assert entropy.shape == (16,)
    print(f"[PASS] GaussianDistribution: sample={samples.shape}, log_prob={log_prob.shape}")

    # Test KL divergence
    old_params = dist.params
    dist.update(mean + 0.1)
    new_params = dist.params
    kl = GaussianDistribution.kl_divergence(old_params, new_params)
    assert kl.shape == (16,)
    assert (kl >= 0).all()
    print(f"[PASS] KL divergence: mean={kl.mean():.6f} (non-negative)")

    # Test rollout storage
    from rsl_rl_ttml.storage.rollout_storage import RolloutStorage

    storage = RolloutStorage(
        num_envs=16,
        num_transitions_per_env=8,
        obs_shapes={"policy": (8,)},
        num_actions=4,
    )

    for step in range(8):
        t = RolloutStorage.Transition()
        t.observations = {"policy": np.random.randn(16, 8).astype(np.float32)}
        t.actions = np.random.randn(16, 4).astype(np.float32)
        t.rewards = np.random.randn(16).astype(np.float32)
        t.dones = np.zeros(16, dtype=np.float32)
        t.values = np.random.randn(16, 1).astype(np.float32)
        t.actions_log_prob = np.random.randn(16).astype(np.float32)
        t.action_mean = np.random.randn(16, 4).astype(np.float32)
        t.action_std = np.ones((16, 4), dtype=np.float32)
        storage.add_transition(t)

    assert storage.step == 8
    print(f"[PASS] RolloutStorage: stored {storage.step} transitions")

    # Test mini-batch generator
    batch_count = 0
    for batch in storage.mini_batch_generator(num_mini_batches=2, num_epochs=1):
        assert batch.observations["policy"].shape[0] == 16 * 8 // 2
        batch_count += 1
    assert batch_count == 2
    print(f"[PASS] Mini-batch generator: {batch_count} batches")

    storage.clear()
    assert storage.step == 0
    print("[PASS] RolloutStorage: clear works")

    print()
    print("All numpy component tests passed!")
    print()


def test_npu_pipeline():
    """Test the full NPU pipeline (requires ttml and Tenstorrent device)."""
    print("=" * 60)
    print("Testing NPU pipeline (requires Tenstorrent device)")
    print("=" * 60)

    try:
        import ttml
        from rsl_rl_ttml.runners.on_policy_runner import OnPolicyRunner

        env = DummyVecEnv(num_envs=64)

        train_cfg = {
            "num_steps_per_env": 8,
            "save_interval": 100,
            "actor_obs_groups": ["policy"],
            "critic_obs_groups": ["policy"],
            "actor": {
                "hidden_dims": (64, 64),
                "activation": "relu",
                "obs_normalization": False,
                "distribution_cfg": {"init_std": 1.0},
            },
            "critic": {
                "hidden_dims": (64, 64),
                "activation": "relu",
                "obs_normalization": False,
            },
            "algorithm": {
                "num_learning_epochs": 2,
                "num_mini_batches": 2,
                "clip_param": 0.2,
                "gamma": 0.99,
                "lam": 0.95,
                "learning_rate": 3e-4,
            },
        }

        runner = OnPolicyRunner(env, train_cfg)
        print("[PASS] OnPolicyRunner: initialized with NPU device")

        # Run a few iterations
        runner.learn(num_learning_iterations=3)
        print("[PASS] OnPolicyRunner: completed 3 training iterations")

        runner.close()
        print("[PASS] NPU device closed successfully")

        print()
        print("All NPU pipeline tests passed!")

    except ImportError as e:
        print(f"[SKIP] ttml not available: {e}")
        print("Run the numpy tests first, then build tt-train to enable NPU tests.")
    except Exception as e:
        print(f"[FAIL] NPU test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_numpy_components()
    test_npu_pipeline()
