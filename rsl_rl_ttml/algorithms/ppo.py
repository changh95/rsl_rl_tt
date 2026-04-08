# SPDX-License-Identifier: BSD-3-Clause

"""PPO algorithm adapted for ttml (Tenstorrent NPU).

Rollout phase: CPU/numpy (env interaction, action sampling, GAE).
Update phase: NPU via ttml (MLP forward/backward, optimizer step).

Architecture:
- Forward passes through actor/critic MLPs run on NPU
- PPO gradient is encoded as regression targets (numpy)
- mse_loss on NPU drives backward pass through MLP weights
- ttml.optimizers.AdamW updates weights on NPU

This approach uses available ttml ops (mse_loss, binary ops) to get
full gradient flow through the MLP while computing PPO-specific terms
(ratio, clipping, log_prob) analytically in numpy.
"""

from __future__ import annotations

import numpy as np

import ttml
import ttnn

from rsl_rl_ttml.models.mlp_model import MLPModel
from rsl_rl_ttml.storage.rollout_storage import RolloutStorage
from rsl_rl_ttml.utils.tensor_utils import numpy_to_ttml, ttml_to_numpy, pad_to_tile


class PPO:
    """Proximal Policy Optimization for ttml."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
    ) -> None:
        self.actor = actor
        self.critic = critic
        self.storage = storage

        # PPO hyperparameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        # Create ttml AdamW optimizer for actor and critic
        all_params = ttml.NamedParameters()
        for name, param in self.actor.parameters():
            all_params[f"actor.{name}"] = param.tensor
        for name, param in self.critic.parameters():
            all_params[f"critic.{name}"] = param.tensor

        opt_config = ttml.optimizers.AdamWConfig.make(
            lr=learning_rate,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            weight_decay=0.0,
        )
        self.optimizer = ttml.optimizers.AdamW(all_params, opt_config)

        # Transition buffer
        self.transition = RolloutStorage.Transition()

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """Sample actions from the actor and store transition data (numpy)."""
        actions = self.actor.act(obs, stochastic=True)
        values = self.critic.get_value(obs)

        self.transition.actions = actions
        self.transition.values = values
        self.transition.actions_log_prob = self.actor.distribution.log_prob(actions)
        self.transition.action_mean = self.actor.distribution.mean.copy()
        self.transition.action_std = self.actor.distribution.std.copy()
        self.transition.observations = obs

        return actions

    def process_env_step(
        self,
        obs: dict[str, np.ndarray],
        rewards: np.ndarray,
        dones: np.ndarray,
        extras: dict,
    ) -> None:
        """Record one environment step."""
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)

        self.transition.rewards = rewards.copy()
        self.transition.dones = dones

        if "time_outs" in extras:
            self.transition.rewards += self.gamma * (
                self.transition.values.squeeze(-1) * extras["time_outs"]
            )

        self.storage.add_transition(self.transition)
        self.transition.clear()

    def compute_returns(self, obs: dict[str, np.ndarray]) -> None:
        """Compute GAE returns and advantages from stored transitions."""
        st = self.storage
        last_values = self.critic.get_value(obs)

        advantage = 0.0
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            next_is_not_terminal = 1.0 - st.dones[step]
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            st.returns[step] = advantage + st.values[step]

        st.advantages = st.returns - st.values
        if not self.normalize_advantage_per_mini_batch:
            adv_mean = st.advantages.mean()
            adv_std = st.advantages.std() + 1e-8
            st.advantages = (st.advantages - adv_mean) / adv_std

    def update(self) -> dict[str, float]:
        """Run PPO optimization epochs on NPU.

        Strategy:
        1. Forward pass through actor/critic MLP on NPU
        2. Compute PPO policy gradient target and value target in numpy
        3. Use mse_loss on NPU to drive backward through MLP weights
        4. Optimizer step updates weights on NPU
        """
        ctx = ttml.autograd.AutoContext.get_instance()

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0

        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )

        for batch in generator:
            if self.normalize_advantage_per_mini_batch:
                batch.advantages = (
                    (batch.advantages - batch.advantages.mean())
                    / (batch.advantages.std() + 1e-8)
                )

            B = batch.actions.shape[0]
            advantages = batch.advantages.squeeze(-1)  # [B]

            # === Actor update on NPU ===
            # Forward pass: get current mean from MLP
            actor_latent = self.actor._get_latent_np(batch.observations)
            actor_input = numpy_to_ttml(actor_latent)
            actor_output = self.actor.forward_ttml(actor_input)
            current_mean = ttml_to_numpy(actor_output, original_shape=(B, self.actor.output_dim))

            # Compute PPO policy gradient target in numpy
            std = self.actor.distribution.std
            var = std ** 2

            # Recompute log probs with current mean
            self.actor.distribution.update(current_mean)
            new_log_prob = self.actor.distribution.log_prob(batch.actions)
            old_log_prob = batch.old_actions_log_prob.squeeze(-1)

            # PPO ratio and clipping
            ratio = np.exp(np.clip(new_log_prob - old_log_prob, -20.0, 20.0))
            clipped_ratio = np.clip(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)

            # Surrogate loss (for logging)
            surr1 = -advantages * ratio
            surr2 = -advantages * clipped_ratio
            surrogate_loss = np.maximum(surr1, surr2).mean()

            # PPO clipping mask: zero gradient when clipping is active
            # Clipping is active when surr2 > surr1 (clipped version is worse)
            clip_mask = (surr1 >= surr2).astype(np.float32)  # [B], 1.0 = unclipped

            # Natural policy gradient direction: advantage * (action - mean) / var
            # Positive advantage -> move mean toward action
            # Negative advantage -> move mean away from action
            d_logprob_d_mean = (batch.actions - current_mean) / var  # [B, A]
            pg_direction = advantages[:, None] * d_logprob_d_mean  # [B, A]

            # Apply PPO clip mask (zero out clipped samples)
            pg_direction *= clip_mask[:, None]

            # Create regression target: nudge mean in the policy gradient direction
            # Small step size to keep updates stable
            pg_step = 0.1
            actor_target = (current_mean + pg_step * pg_direction).astype(np.float32)
            actor_target_ttml = numpy_to_ttml(actor_target)

            # Compute actor loss on NPU (mse_loss with backward)
            self.optimizer.zero_grad()
            actor_loss = ttml.ops.loss.mse_loss(actor_output, actor_target_ttml, ttml.ops.ReduceType.MEAN)

            # === Critic update on NPU ===
            critic_latent = self.critic._get_latent_np(batch.observations)
            critic_input = numpy_to_ttml(critic_latent)
            critic_output = self.critic.forward_ttml(critic_input)

            # Value target from GAE returns
            value_target = numpy_to_ttml(batch.returns.astype(np.float32))
            critic_loss = ttml.ops.loss.mse_loss(critic_output, value_target, ttml.ops.ReduceType.MEAN)

            # Combined loss: we need to backward both
            # Since they share the optimizer, we can backward sequentially
            actor_loss.backward(False)
            critic_loss.backward(False)
            self.optimizer.step()
            ctx.reset_graph()

            # Entropy (computed in numpy for logging)
            entropy_mean = self.actor.distribution.entropy().mean()

            # Compute value loss for logging
            current_values = ttml_to_numpy(critic_output, original_shape=(B, 1))
            value_loss = ((current_values - batch.returns) ** 2).mean()

            # === Update std (entropy) via REINFORCE gradient ===
            if self.actor.distribution is not None:
                self.actor.distribution.update_std(
                    batch.actions, advantages, lr=self.learning_rate * 0.5
                )

            # === Adaptive LR ===
            if self.desired_kl is not None and self.schedule == "adaptive":
                new_params = (current_mean, std.copy())
                old_params = (batch.old_action_mean, batch.old_action_std)
                kl = self.actor.distribution.kl_divergence(old_params, new_params)
                kl_mean = kl.mean()

                if kl_mean > self.desired_kl * 2.0:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                self.optimizer.set_lr(self.learning_rate)

            mean_value_loss += value_loss
            mean_surrogate_loss += surrogate_loss
            mean_entropy += entropy_mean

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates

        self.storage.clear()

        return {
            "value": float(mean_value_loss),
            "surrogate": float(mean_surrogate_loss),
            "entropy": float(mean_entropy),
        }

    def train_mode(self) -> None:
        self.actor.train()
        self.critic.train()

    def eval_mode(self) -> None:
        self.actor.eval()
        self.critic.eval()

    def get_policy(self) -> MLPModel:
        return self.actor

    def save(self) -> dict:
        return {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
        }

    def load(self, loaded_dict: dict) -> None:
        if "actor_state_dict" in loaded_dict:
            self.actor.load_state_dict(loaded_dict["actor_state_dict"])
        if "critic_state_dict" in loaded_dict:
            self.critic.load_state_dict(loaded_dict["critic_state_dict"])
