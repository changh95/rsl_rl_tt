#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause

"""Large workload comparison: CPU vs TTML NPU.

Runs identical PPO training on both backends with increasing workload sizes
and reports throughput (FPS), learning quality, and wall-clock time.

Usage:
    source ~/tt-metal/python_env/bin/activate
    PYTHONPATH="~/tt-metal/build_Release/ttml:~/experiments/rl-rsl/rsl_rl_tt:$PYTHONPATH" \
        python tests/test_large_workload.py
"""

from __future__ import annotations

import copy
import time
import torch
import numpy as np
from tensordict import TensorDict
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.env.vec_env import VecEnv


class ScalableEnv(VecEnv):
    """Environment with configurable observation and action dimensions."""

    def __init__(self, num_envs: int = 256, obs_dim: int = 48, act_dim: int = 12) -> None:
        self.num_envs = num_envs
        self.num_actions = act_dim
        self.max_episode_length = 200
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        self.device = "cpu"
        self.cfg = {}
        self._obs_dim = obs_dim
        self._state = torch.randn(num_envs, obs_dim) * 0.1

    def get_observations(self) -> TensorDict:
        return TensorDict({"policy": self._state.clone()}, batch_size=[self.num_envs])

    def step(self, actions: torch.Tensor) -> tuple:
        # Simple dynamics
        self._state[:, : self.num_actions] += actions * 0.05
        self._state *= 0.99  # decay
        self._state += torch.randn_like(self._state) * 0.01

        rewards = -torch.norm(self._state, dim=-1)
        rewards += (torch.norm(self._state[:, :3], dim=-1) < 0.2).float() * 2.0

        self.episode_length_buf += 1
        dones = (self.episode_length_buf >= self.max_episode_length).float()
        time_outs = dones.clone()
        done_mask = dones > 0.5
        if done_mask.any():
            self._state[done_mask] = torch.randn(done_mask.sum(), self._obs_dim) * 0.1
            self.episode_length_buf[done_mask] = 0

        return self.get_observations(), rewards, dones, {"time_outs": time_outs}


def make_cfg(hidden_dims: tuple, num_steps: int = 24) -> dict:
    return {
        "num_steps_per_env": num_steps,
        "save_interval": 99999,
        "check_for_nan": False,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": hidden_dims,
            "activation": "relu",
            "obs_normalization": False,
            "distribution_cfg": {
                "class_name": "rsl_rl.modules.distribution:GaussianDistribution",
                "init_std": 1.0,
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": hidden_dims,
            "activation": "relu",
            "obs_normalization": False,
        },
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 4,
            "num_mini_batches": 4,
            "clip_param": 0.2,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 3e-4,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
    }


def run_timed(device: str, env: VecEnv, cfg: dict, duration_sec: float) -> dict:
    """Run training for a fixed duration, return stats."""
    runner = OnPolicyRunner(env, cfg, device=device)

    iters = 0
    total_steps = 0
    num_steps = cfg["num_steps_per_env"]
    losses = []

    t_start = time.time()
    while time.time() - t_start < duration_sec:
        runner.learn(num_learning_iterations=1)
        iters += 1
        total_steps += num_steps * env.num_envs

    elapsed = time.time() - t_start
    fps = total_steps / elapsed

    # Get final loss
    obs = env.get_observations().to("cpu" if device.startswith("ttml") else device)
    with torch.inference_mode():
        for _ in range(num_steps):
            a = runner.alg.act(obs)
            obs, rw, dn, ex = env.step(a)
            runner.alg.process_env_step(obs, rw, dn, ex)
        runner.alg.compute_returns(obs)
    final_loss = runner.alg.update()

    if hasattr(runner, 'close'):
        runner.close()

    return {
        "device": device,
        "iters": iters,
        "total_steps": total_steps,
        "elapsed": elapsed,
        "fps": fps,
        "value_loss": final_loss["value"],
        "surrogate_loss": final_loss["surrogate"],
    }


def run_workload(label: str, num_envs: int, obs_dim: int, act_dim: int,
                 hidden_dims: tuple, duration: float = 60.0) -> dict:
    """Run one workload on both CPU and TTML, return comparison."""
    print(f"\n{'=' * 70}")
    print(f"WORKLOAD: {label}")
    print(f"  envs={num_envs}, obs={obs_dim}, act={act_dim}, hidden={hidden_dims}")
    print(f"  Duration: {duration:.0f}s per device")
    print(f"{'=' * 70}")

    cfg = make_cfg(hidden_dims)

    # CPU
    print(f"\n  Running CPU...")
    torch.manual_seed(42)
    env_cpu = ScalableEnv(num_envs, obs_dim, act_dim)
    cpu_result = run_timed("cpu", env_cpu, copy.deepcopy(cfg), duration)
    print(f"  CPU: {cpu_result['iters']} iters, {cpu_result['fps']:.0f} FPS, "
          f"value_loss={cpu_result['value_loss']:.4f}")

    # TTML
    print(f"  Running TTML...")
    torch.manual_seed(42)
    env_tt = ScalableEnv(num_envs, obs_dim, act_dim)
    tt_result = run_timed("ttml", env_tt, copy.deepcopy(cfg), duration)
    print(f"  TTML: {tt_result['iters']} iters, {tt_result['fps']:.0f} FPS, "
          f"value_loss={tt_result['value_loss']:.4f}")

    speedup = tt_result["fps"] / cpu_result["fps"]
    print(f"\n  Speedup: {speedup:.2f}x ({'TTML faster' if speedup > 1 else 'CPU faster'})")
    print(f"  Iters:   CPU={cpu_result['iters']}, TTML={tt_result['iters']}")

    return {"label": label, "cpu": cpu_result, "ttml": tt_result, "speedup": speedup}


if __name__ == "__main__":
    print("=" * 70)
    print("LARGE WORKLOAD TEST: CPU vs TTML NPU")
    print("=" * 70)

    results = []

    # Workload 1: Small (typical robot RL - similar to what we had before)
    results.append(run_workload(
        "Small (robot-like)",
        num_envs=256, obs_dim=48, act_dim=12,
        hidden_dims=(128, 128, 128),
        duration=30.0,
    ))

    # Workload 2: Medium (larger network)
    results.append(run_workload(
        "Medium (larger net)",
        num_envs=512, obs_dim=64, act_dim=16,
        hidden_dims=(256, 256, 256),
        duration=60.0,
    ))

    # Workload 3: Large (big batch + big network)
    results.append(run_workload(
        "Large (big batch+net)",
        num_envs=1024, obs_dim=128, act_dim=32,
        hidden_dims=(512, 512, 512),
        duration=60.0,
    ))

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Workload':<25} {'CPU FPS':>10} {'TTML FPS':>10} {'Speedup':>10} {'CPU VLoss':>10} {'TTML VLoss':>10}")
    print("-" * 75)
    for r in results:
        print(f"{r['label']:<25} {r['cpu']['fps']:>10.0f} {r['ttml']['fps']:>10.0f} "
              f"{r['speedup']:>9.2f}x {r['cpu']['value_loss']:>10.4f} {r['ttml']['value_loss']:>10.4f}")

    print(f"\nAll workloads completed.")
