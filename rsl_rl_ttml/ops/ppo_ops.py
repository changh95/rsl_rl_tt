# SPDX-License-Identifier: BSD-3-Clause

"""Custom ttml autograd functions for PPO loss computation.

These implement the missing ops (exp, clamp) needed for a proper PPO
surrogate loss, with both forward and backward passes on the NPU.
"""

import ttnn
import ttml


class Exp(ttml.autograd.Function):
    """Elementwise exp with autograd support.

    forward: y = exp(x)
    backward: d(loss)/d(x) = d(loss)/d(y) * exp(x) = grad_output * y
    """

    @staticmethod
    def forward(ctx, x):
        y = ttnn.exp(x.get_value())
        ctx.save_for_backward(None)  # We'll save y instead
        ctx._y = y  # Save output for backward
        return y

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx._y
        return ttnn.multiply(grad_output, y)


class Clamp(ttml.autograd.Function):
    """Elementwise clamp with autograd support.

    forward: y = clamp(x, min_val, max_val)
    backward: d(loss)/d(x) = grad_output * (min_val < x < max_val)
              (gradient passes through where not clamped, zero where clamped)
    """

    @staticmethod
    def forward(ctx, x, min_val, max_val):
        x_val = x.get_value()
        # clamp = max(min_val, min(x, max_val))
        y = ttnn.clip(x_val, min=min_val, max=max_val)
        # Save mask for backward: 1 where not clamped, 0 where clamped
        # not_clamped = (x > min_val) & (x < max_val)
        above_min = ttnn.gt(x_val, min_val)
        below_max = ttnn.lt(x_val, max_val)
        ctx._mask = ttnn.multiply(above_min, below_max)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        return ttnn.multiply(grad_output, ctx._mask)


class SquaredDiff(ttml.autograd.Function):
    """Compute (a - b)^2 with autograd on a only (b is treated as constant).

    forward: y = (a - target)^2
    backward: d(loss)/d(a) = grad_output * 2 * (a - target)
    """

    @staticmethod
    def forward(ctx, a, target_val):
        a_val = a.get_value()
        diff = ttnn.subtract(a_val, target_val)
        ctx._diff = diff
        return ttnn.multiply(diff, diff)

    @staticmethod
    def backward(ctx, grad_output):
        return ttnn.multiply(grad_output, ttnn.multiply(ctx._diff, 2.0))


class SurrogateLoss(ttml.autograd.Function):
    """PPO surrogate loss as a single fused operation.

    Given actor_output (mean) from MLP, compute the full PPO surrogate loss
    with gradients flowing back through the mean.

    forward:
        1. log_prob = -0.5 * sum((action - mean)^2 / var)  (ignoring constants)
        2. ratio = exp(log_prob - old_log_prob)
        3. clipped_ratio = clamp(ratio, 1-eps, 1+eps)
        4. loss = max(-adv * ratio, -adv * clipped_ratio)

    backward:
        d(loss)/d(mean) = d(loss)/d(ratio) * d(ratio)/d(log_prob) * d(log_prob)/d(mean)
        where d(log_prob)/d(mean) = (action - mean) / var
    """

    @staticmethod
    def forward(ctx, actor_output, actions_np, old_log_prob_np, advantages_np, std_np, clip_param):
        """Compute PPO surrogate loss.

        Args:
            actor_output: ttml Tensor [1, 1, B_pad, D_pad] - MLP output (mean)
            actions_np: numpy [B, A] - sampled actions
            old_log_prob_np: numpy [B] - old log probabilities
            advantages_np: numpy [B] - advantages
            std_np: numpy [A] - current std
            clip_param: float - PPO clip epsilon
        """
        import numpy as np
        from rsl_rl_ttml.utils.tensor_utils import ttml_to_numpy, pad_to_tile

        # Get current mean from NPU output
        mean_val = actor_output.get_value()
        B_pad = mean_val.shape[-2]
        D_pad = mean_val.shape[-1]
        mean_np_full = actor_output.to_numpy(ttnn.DataType.FLOAT32).reshape(B_pad, D_pad)

        B = actions_np.shape[0]
        A = actions_np.shape[1]
        mean_np = mean_np_full[:B, :A]

        var = std_np ** 2

        # Compute log prob
        log_prob = -0.5 * np.sum((actions_np - mean_np) ** 2 / var, axis=-1)
        # Remove constants (they cancel in ratio)

        # Ratio
        ratio = np.exp(np.clip(log_prob - old_log_prob_np, -20.0, 20.0))
        clipped_ratio = np.clip(ratio, 1.0 - clip_param, 1.0 + clip_param)

        # Surrogate
        surr1 = -advantages_np * ratio
        surr2 = -advantages_np * clipped_ratio
        loss_per_sample = np.maximum(surr1, surr2)  # [B]

        # Compute gradient d(loss)/d(mean)
        use_clipped = (surr2 > surr1).astype(np.float32)
        effective_ratio = ratio * (1.0 - use_clipped) + clipped_ratio * use_clipped

        # d(loss)/d(mean) via chain rule
        # d(surrogate)/d(ratio) = -advantage (when not clipped)
        # d(ratio)/d(log_prob) = ratio
        # d(log_prob)/d(mean) = (action - mean) / var
        # Combined: -advantage * ratio * (action - mean) / var (when not clipped, 0 when clipped)
        d_logprob_d_mean = (actions_np - mean_np) / var  # [B, A]
        d_loss_d_mean = (-advantages_np * effective_ratio)[:, None] * d_logprob_d_mean  # [B, A]

        # Store gradient for backward - pad to tile alignment
        grad_padded = np.zeros((B_pad, D_pad), dtype=np.float32)
        grad_padded[:B, :A] = d_loss_d_mean / B  # normalize by batch size

        ctx._grad_for_mean = grad_padded
        ctx._loss_value = float(loss_per_sample.mean())

        # Return a scalar-ish loss tensor (for logging, not for backward directly)
        loss_arr = np.array([[[[ctx._loss_value]]]], dtype=np.float32)
        return ttml.autograd.Tensor.from_numpy(loss_arr).get_value()

    @staticmethod
    def backward(ctx, grad_output):
        import numpy as np
        # Return the pre-computed gradient for the actor_output
        grad_np = ctx._grad_for_mean.reshape(1, 1, *ctx._grad_for_mean.shape)
        grad_tensor = ttml.autograd.Tensor.from_numpy(grad_np.astype(np.float32), layout=ttnn.Layout.TILE)
        return grad_tensor.get_value()
