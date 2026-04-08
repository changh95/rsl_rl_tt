# SPDX-License-Identifier: BSD-3-Clause

"""Test environments for validating PPO training without a simulator.

Each environment has a known optimal policy so we can verify convergence.
"""

from __future__ import annotations

import numpy as np
from rsl_rl_ttml.env.vec_env import VecEnv


class ReachTargetEnv(VecEnv):
    """Move a point mass to the origin in 2D. Known optimal: action = -k * position.

    State: [x, y, vx, vy] (position + velocity)
    Action: [ax, ay] (acceleration, clipped to [-1, 1])
    Reward: -distance_to_origin (negative L2 norm of position)
    Optimal policy: proportional control toward origin.
    """

    def __init__(self, num_envs: int = 128) -> None:
        self.num_envs = num_envs
        self.num_actions = 2
        self.max_episode_length = 200
        self.episode_length_buf = np.zeros(num_envs, dtype=np.int32)
        self.dt = 0.05

        # State: [x, y, vx, vy]
        self._state = np.zeros((num_envs, 4), dtype=np.float32)
        self._reset_all()

    def _reset_all(self):
        self._state[:, :2] = np.random.uniform(-1.0, 1.0, (self.num_envs, 2)).astype(np.float32)
        self._state[:, 2:] = 0.0  # zero initial velocity

    def _reset_envs(self, mask):
        n = mask.sum()
        self._state[mask, :2] = np.random.uniform(-1.0, 1.0, (n, 2)).astype(np.float32)
        self._state[mask, 2:] = 0.0
        self.episode_length_buf[mask] = 0

    def get_observations(self) -> dict[str, np.ndarray]:
        return {"policy": self._state.copy()}

    def step(self, actions: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict]:
        actions = np.clip(actions, -1.0, 1.0)

        # Simple point-mass dynamics
        self._state[:, 2:] += actions * self.dt  # velocity += accel * dt
        self._state[:, 2:] *= 0.95  # damping
        self._state[:, :2] += self._state[:, 2:] * self.dt  # position += vel * dt

        # Reward: negative distance from origin
        dist = np.linalg.norm(self._state[:, :2], axis=-1)
        rewards = -dist.astype(np.float32)

        # Bonus for being close
        rewards += (dist < 0.1).astype(np.float32) * 1.0

        self.episode_length_buf += 1
        dones = (self.episode_length_buf >= self.max_episode_length).astype(np.float32)
        time_outs = dones.copy()

        done_mask = dones > 0.5
        if done_mask.any():
            self._reset_envs(done_mask)

        return self.get_observations(), rewards, dones, {"time_outs": time_outs}


class BanditEnv(VecEnv):
    """N-armed bandit. Known optimal: always pick the arm with highest mean.

    State: constant [1.0] (contextless bandit)
    Action: continuous in [-1, 1]^N, reward = -|action - target|
    Target is fixed per episode. Optimal: action = target.

    This tests whether the actor can learn a fixed mapping.
    """

    def __init__(self, num_envs: int = 128, num_actions: int = 4) -> None:
        self.num_envs = num_envs
        self.num_actions = num_actions
        self.max_episode_length = 50
        self.episode_length_buf = np.zeros(num_envs, dtype=np.int32)

        # Fixed target the agent should learn to output
        self._target = np.array([0.5, -0.3, 0.1, -0.7][:num_actions], dtype=np.float32)

    def get_observations(self) -> dict[str, np.ndarray]:
        # Constant observation (contextless)
        return {"policy": np.ones((self.num_envs, 4), dtype=np.float32)}

    def step(self, actions: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict]:
        # Reward = negative L2 distance from target
        error = actions - self._target
        rewards = -np.sum(error ** 2, axis=-1).astype(np.float32)

        self.episode_length_buf += 1
        dones = (self.episode_length_buf >= self.max_episode_length).astype(np.float32)

        done_mask = dones > 0.5
        self.episode_length_buf[done_mask] = 0

        return self.get_observations(), rewards, dones, {"time_outs": dones.copy()}


class GymVecEnv(VecEnv):
    """Wraps Gymnasium environments into our numpy VecEnv interface.

    Runs N independent Gym envs in parallel (synchronous).

    Usage:
        env = GymVecEnv("Pendulum-v1", num_envs=64)
    """

    def __init__(self, env_id: str, num_envs: int = 64) -> None:
        import gymnasium as gym

        self.num_envs = num_envs
        self._envs = [gym.make(env_id) for _ in range(num_envs)]

        # Infer dimensions from first env
        sample_env = self._envs[0]
        obs_space = sample_env.observation_space
        act_space = sample_env.action_space

        self.num_actions = act_space.shape[0] if hasattr(act_space, 'shape') else 1
        self.max_episode_length = sample_env.spec.max_episode_steps or 200
        self.episode_length_buf = np.zeros(num_envs, dtype=np.int32)

        self._obs_dim = obs_space.shape[0]
        self._obs = np.zeros((num_envs, self._obs_dim), dtype=np.float32)

        # Reset all envs
        for i, env in enumerate(self._envs):
            obs, _ = env.reset()
            self._obs[i] = obs

    def get_observations(self) -> dict[str, np.ndarray]:
        return {"policy": self._obs.copy()}

    def step(self, actions: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict]:
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=np.float32)
        time_outs = np.zeros(self.num_envs, dtype=np.float32)

        for i, env in enumerate(self._envs):
            action = actions[i]
            obs, reward, terminated, truncated, info = env.step(action)
            self._obs[i] = obs
            rewards[i] = reward
            self.episode_length_buf[i] += 1

            if terminated or truncated:
                dones[i] = 1.0
                if truncated and not terminated:
                    time_outs[i] = 1.0
                obs, _ = env.reset()
                self._obs[i] = obs
                self.episode_length_buf[i] = 0

        return self.get_observations(), rewards, dones, {"time_outs": time_outs}
