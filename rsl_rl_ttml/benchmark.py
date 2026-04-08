#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark PPO training on Tenstorrent NPU across multiple test environments.

Tests convergence on environments with known optimal behavior:
1. ReachTarget: point-mass to origin (tests continuous control)
2. Bandit: learn a fixed target output (tests basic optimization)
3. Gymnasium Pendulum: classic control benchmark

Usage:
    source ~/tt-metal/python_env/bin/activate
    PYTHONPATH="~/tt-metal/build_Release/ttml:~/experiments/rl-rsl:$PYTHONPATH" \
        python -m rsl_rl_ttml.benchmark
"""

from __future__ import annotations

import time
import numpy as np


def make_train_cfg(
    num_steps: int = 16,
    hidden_dims: tuple = (64, 64),
    lr: float = 3e-4,
    num_epochs: int = 4,
    num_mini_batches: int = 4,
) -> dict:
    return {
        "num_steps_per_env": num_steps,
        "save_interval": 9999,
        "actor_obs_groups": ["policy"],
        "critic_obs_groups": ["policy"],
        "actor": {
            "hidden_dims": hidden_dims,
            "activation": "relu",
            "obs_normalization": False,
            "distribution_cfg": {"init_std": 1.0},
        },
        "critic": {
            "hidden_dims": hidden_dims,
            "activation": "relu",
            "obs_normalization": False,
        },
        "algorithm": {
            "num_learning_epochs": num_epochs,
            "num_mini_batches": num_mini_batches,
            "clip_param": 0.2,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": lr,
            "schedule": "adaptive",
            "desired_kl": 0.01,
        },
    }


def evaluate_policy(runner, env, num_episodes: int = 20) -> dict:
    """Evaluate the current policy without training."""
    episode_rewards = []
    episode_lengths = []
    obs = env.get_observations()
    ep_reward = np.zeros(env.num_envs, dtype=np.float32)
    ep_len = np.zeros(env.num_envs, dtype=np.int32)

    steps = 0
    max_steps = num_episodes * env.max_episode_length // env.num_envs + env.max_episode_length

    while len(episode_rewards) < num_episodes and steps < max_steps:
        actions = runner.alg.actor.act(obs, stochastic=False)  # deterministic
        obs, rewards, dones, _ = env.step(actions)
        ep_reward += rewards
        ep_len += 1
        steps += 1

        for i in range(env.num_envs):
            if dones[i] > 0.5:
                episode_rewards.append(ep_reward[i])
                episode_lengths.append(ep_len[i])
                ep_reward[i] = 0.0
                ep_len[i] = 0

    if not episode_rewards:
        # Force collect partial episodes
        episode_rewards = list(ep_reward[:num_episodes])
        episode_lengths = list(ep_len[:num_episodes])

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "num_episodes": len(episode_rewards),
    }


def benchmark_reach_target(num_iters: int = 100):
    """Test: can the agent learn to move toward the origin?"""
    from rsl_rl_ttml.envs.test_envs import ReachTargetEnv
    from rsl_rl_ttml.runners.on_policy_runner import OnPolicyRunner

    print("=" * 70)
    print("BENCHMARK: ReachTarget (point-mass to origin)")
    print("=" * 70)

    env = ReachTargetEnv(num_envs=128)
    cfg = make_train_cfg(num_steps=32, lr=3e-4)
    runner = OnPolicyRunner(env, cfg)

    # Evaluate before training
    pre_eval = evaluate_policy(runner, env)
    print(f"Before training: reward={pre_eval['mean_reward']:.2f} +/- {pre_eval['std_reward']:.2f}")

    # Train
    t0 = time.time()
    runner.learn(num_learning_iterations=num_iters)
    train_time = time.time() - t0

    # Evaluate after training
    post_eval = evaluate_policy(runner, env)
    print(f"After training:  reward={post_eval['mean_reward']:.2f} +/- {post_eval['std_reward']:.2f}")

    improved = post_eval["mean_reward"] > pre_eval["mean_reward"]
    print(f"Improved: {'YES' if improved else 'NO'} (delta={post_eval['mean_reward'] - pre_eval['mean_reward']:.2f})")
    print(f"Train time: {train_time:.1f}s ({num_iters} iters)")

    runner.close()
    return improved, pre_eval, post_eval


def benchmark_bandit(num_iters: int = 100):
    """Test: can the agent learn a fixed target output?"""
    from rsl_rl_ttml.envs.test_envs import BanditEnv
    from rsl_rl_ttml.runners.on_policy_runner import OnPolicyRunner

    print()
    print("=" * 70)
    print("BENCHMARK: Bandit (learn fixed target actions)")
    print("=" * 70)

    env = BanditEnv(num_envs=128, num_actions=4)
    cfg = make_train_cfg(num_steps=16, lr=3e-4)
    runner = OnPolicyRunner(env, cfg)

    pre_eval = evaluate_policy(runner, env)
    print(f"Before training: reward={pre_eval['mean_reward']:.4f}")

    t0 = time.time()
    runner.learn(num_learning_iterations=num_iters)
    train_time = time.time() - t0

    post_eval = evaluate_policy(runner, env)
    print(f"After training:  reward={post_eval['mean_reward']:.4f}")

    # Check what the policy actually outputs
    obs = env.get_observations()
    learned_action = runner.alg.actor.act(obs, stochastic=False)[0]
    print(f"Target:          {env._target}")
    print(f"Learned action:  {learned_action}")
    print(f"Action error:    {np.abs(learned_action - env._target).mean():.4f}")

    improved = post_eval["mean_reward"] > pre_eval["mean_reward"]
    print(f"Improved: {'YES' if improved else 'NO'}")
    print(f"Train time: {train_time:.1f}s ({num_iters} iters)")

    runner.close()
    return improved, pre_eval, post_eval


def benchmark_pendulum(num_iters: int = 150):
    """Test: classic Pendulum-v1 from Gymnasium."""
    from rsl_rl_ttml.envs.test_envs import GymVecEnv
    from rsl_rl_ttml.runners.on_policy_runner import OnPolicyRunner

    print()
    print("=" * 70)
    print("BENCHMARK: Gymnasium Pendulum-v1")
    print("=" * 70)

    env = GymVecEnv("Pendulum-v1", num_envs=64)
    cfg = make_train_cfg(num_steps=32, hidden_dims=(64, 64), lr=1e-3)
    runner = OnPolicyRunner(env, cfg)

    pre_eval = evaluate_policy(runner, env)
    print(f"Before training: reward={pre_eval['mean_reward']:.2f}")

    t0 = time.time()
    runner.learn(num_learning_iterations=num_iters)
    train_time = time.time() - t0

    post_eval = evaluate_policy(runner, env)
    print(f"After training:  reward={post_eval['mean_reward']:.2f}")

    improved = post_eval["mean_reward"] > pre_eval["mean_reward"]
    print(f"Improved: {'YES' if improved else 'NO'} (delta={post_eval['mean_reward'] - pre_eval['mean_reward']:.2f})")
    print(f"Train time: {train_time:.1f}s ({num_iters} iters)")

    runner.close()
    return improved, pre_eval, post_eval


if __name__ == "__main__":
    results = {}

    r1 = benchmark_reach_target(num_iters=100)
    results["reach_target"] = r1[0]

    r2 = benchmark_bandit(num_iters=100)
    results["bandit"] = r2[0]

    r3 = benchmark_pendulum(num_iters=150)
    results["pendulum"] = r3[0]

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
    print()

    total_pass = sum(results.values())
    print(f"Result: {total_pass}/{len(results)} benchmarks showed improvement")
