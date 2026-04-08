# SPDX-License-Identifier: BSD-3-Clause

"""On-policy runner using ttml AutoContext for Tenstorrent NPU.

The runner orchestrates:
1. Environment interaction (CPU/numpy)
2. Rollout data collection (CPU/numpy)
3. PPO updates with NPU-accelerated forward/backward passes
"""

from __future__ import annotations

import os
import time
import pickle
import numpy as np

import ttml

from rsl_rl_ttml.algorithms.ppo import PPO
from rsl_rl_ttml.env.vec_env import VecEnv
from rsl_rl_ttml.models.mlp_model import MLPModel
from rsl_rl_ttml.storage.rollout_storage import RolloutStorage


class OnPolicyRunner:
    """On-policy RL runner for Tenstorrent NPU."""

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
    ) -> None:
        self.env = env
        self.cfg = train_cfg
        self.log_dir = log_dir

        # Initialize ttml device
        self.ctx = ttml.autograd.AutoContext.get_instance()
        self.ctx.open_device()
        print("Tenstorrent NPU device opened.")

        # Get initial observations to determine shapes
        obs = self.env.get_observations()
        obs_dims = {key: val.shape[-1] for key, val in obs.items()}
        obs_shapes = {key: (val.shape[-1],) for key, val in obs.items()}

        # Resolve observation groups
        actor_obs_groups = train_cfg.get("actor_obs_groups", list(obs.keys()))
        critic_obs_groups = train_cfg.get("critic_obs_groups", list(obs.keys()))

        # Create actor and critic
        actor_cfg = train_cfg.get("actor", {})
        critic_cfg = train_cfg.get("critic", {})

        self.actor = MLPModel(
            obs_groups=actor_obs_groups,
            obs_dims=obs_dims,
            obs_set="actor",
            output_dim=env.num_actions,
            hidden_dims=actor_cfg.get("hidden_dims", (256, 256, 256)),
            activation=actor_cfg.get("activation", "elu"),
            obs_normalization=actor_cfg.get("obs_normalization", False),
            distribution_cfg=actor_cfg.get("distribution_cfg", {"init_std": 1.0}),
        )

        self.critic = MLPModel(
            obs_groups=critic_obs_groups,
            obs_dims=obs_dims,
            obs_set="critic",
            output_dim=1,
            hidden_dims=critic_cfg.get("hidden_dims", (256, 256, 256)),
            activation=critic_cfg.get("activation", "elu"),
            obs_normalization=critic_cfg.get("obs_normalization", False),
            distribution_cfg=None,  # Critic has no distribution
        )

        # Create storage
        num_steps = train_cfg.get("num_steps_per_env", 24)
        self.storage = RolloutStorage(
            num_envs=env.num_envs,
            num_transitions_per_env=num_steps,
            obs_shapes=obs_shapes,
            num_actions=env.num_actions,
        )

        # Create PPO algorithm
        alg_cfg = train_cfg.get("algorithm", {})
        self.alg = PPO(
            actor=self.actor,
            critic=self.critic,
            storage=self.storage,
            num_learning_epochs=alg_cfg.get("num_learning_epochs", 5),
            num_mini_batches=alg_cfg.get("num_mini_batches", 4),
            clip_param=alg_cfg.get("clip_param", 0.2),
            gamma=alg_cfg.get("gamma", 0.99),
            lam=alg_cfg.get("lam", 0.95),
            value_loss_coef=alg_cfg.get("value_loss_coef", 1.0),
            entropy_coef=alg_cfg.get("entropy_coef", 0.01),
            learning_rate=alg_cfg.get("learning_rate", 0.001),
            max_grad_norm=alg_cfg.get("max_grad_norm", 1.0),
            schedule=alg_cfg.get("schedule", "adaptive"),
            desired_kl=alg_cfg.get("desired_kl", 0.01),
        )

        self.current_learning_iteration = 0

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Run the training loop."""
        if init_at_random_ep_len:
            self.env.episode_length_buf = np.random.randint(
                0, int(self.env.max_episode_length), size=self.env.num_envs
            )

        obs = self.env.get_observations()
        self.alg.train_mode()

        num_steps = self.cfg.get("num_steps_per_env", 24)
        save_interval = self.cfg.get("save_interval", 50)

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations

        for it in range(start_it, total_it):
            start = time.time()

            # === Rollout Phase (CPU/numpy) ===
            for _ in range(num_steps):
                actions = self.alg.act(obs)
                obs, rewards, dones, extras = self.env.step(actions)
                self.alg.process_env_step(obs, rewards, dones, extras)

            collect_time = time.time() - start
            start = time.time()

            # Compute returns (CPU/numpy)
            self.alg.compute_returns(obs)

            # === Update Phase (NPU via ttml) ===
            loss_dict = self.alg.update()

            learn_time = time.time() - start
            self.current_learning_iteration = it

            # Log
            fps = num_steps * self.env.num_envs / (collect_time + learn_time)
            print(
                f"Iter {it}/{total_it} | "
                f"surrogate: {loss_dict['surrogate']:.4f} | "
                f"value: {loss_dict['value']:.4f} | "
                f"entropy: {loss_dict['entropy']:.4f} | "
                f"collect: {collect_time:.2f}s | "
                f"learn: {learn_time:.2f}s | "
                f"FPS: {fps:.0f}"
            )

            # Save
            if self.log_dir and it % save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pkl"))

        # Final save
        if self.log_dir:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pkl"))

    def save(self, path: str) -> None:
        """Save model to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        with open(path, "wb") as f:
            pickle.dump(saved_dict, f)
        print(f"Saved model to {path}")

    def load(self, path: str) -> None:
        """Load model from file."""
        with open(path, "rb") as f:
            loaded_dict = pickle.load(f)
        self.alg.load(loaded_dict)
        self.current_learning_iteration = loaded_dict.get("iter", 0)
        print(f"Loaded model from {path} (iter {self.current_learning_iteration})")

    def close(self) -> None:
        """Close the ttml device."""
        self.ctx.close_device()
        print("Tenstorrent NPU device closed.")
