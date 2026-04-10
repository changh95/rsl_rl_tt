# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom ttml autograd operators for PPO loss computation on NPU.

Reference: https://github.com/tenstorrent/tt-metal/blob/3e71a088/tt-train/sources/examples/grpo/utils/ttml_operators.py
"""

import ttnn
import ttml
from ttml.autograd import Function


class Exp(Function):
    @staticmethod
    def forward(ctx, x):
        result = ttnn.exp(x.get_value())
        out = ttml.autograd.create_tensor(result)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (exp_x,) = ctx.saved_tensors
        return ttnn.multiply(grad_output, exp_x.get_value())


class Clip(Function):
    @staticmethod
    def forward(ctx, x, lo, hi):
        val = x.get_value()
        clipped = ttnn.clip(val, lo, hi)
        mask = ttnn.multiply(ttnn.ge(val, lo), ttnn.le(val, hi))
        ctx.mask = mask
        return clipped

    @staticmethod
    def backward(ctx, grad_output):
        return ttnn.multiply(grad_output, ctx.mask)


class Min(Function):
    @staticmethod
    def forward(ctx, a, b):
        a_val = a.get_value()
        b_val = b.get_value()
        result = ttnn.minimum(a_val, b_val)
        a_wins = ttnn.le(a_val, b_val)
        ctx.a_wins = a_wins
        return result

    @staticmethod
    def backward(ctx, grad_output):
        a_wins = ctx.a_wins
        b_wins = ttnn.subtract(ttnn.ones_like(a_wins), a_wins)
        grad_a = ttnn.multiply(grad_output, a_wins)
        grad_b = ttnn.multiply(grad_output, b_wins)
        return grad_a, grad_b


class FusedPPOLoss(Function):
    """Fused PPO surrogate loss: all ops in one forward/backward call.

    Computes the full PPO clipped surrogate loss from actor output (mean)
    using raw ttnn ops, avoiding per-op autograd graph construction overhead.

    Forward:
        diff = actions - mean
        log_prob = -0.5 * mean(diff^2 / var) * A
        ratio = exp(clip(log_prob - old_log_prob, -20, 20))
        surr1 = -advantage * ratio
        surr2 = -advantage * clip(ratio, 1-eps, 1+eps)
        loss = mean(max(surr1, surr2))

    Backward:
        d(loss)/d(mean) via chain rule through all ops.
    """

    @staticmethod
    def forward(ctx, mean_ttml, actions_ttml, old_logp_ttml, adv_ttml, var_ttml, clip_eps, num_actions):
        m = mean_ttml.get_value()
        a = actions_ttml.get_value()
        olp = old_logp_ttml.get_value()
        adv = adv_ttml.get_value()
        var = var_ttml.get_value()
        A = float(num_actions)

        # log_prob = -0.5 * mean(diff^2 / var) * A
        diff = ttnn.subtract(a, m)
        diff_sq = ttnn.multiply(diff, diff)
        scaled = ttnn.multiply(ttnn.divide(diff_sq, var), -0.5)
        # mean over all dims, scale by A
        new_logp = ttnn.multiply(ttnn.mean(scaled), A)

        # ratio = exp(clip(new_logp - old_logp, -20, 20))
        logp_diff = ttnn.subtract(new_logp, olp)
        logp_clipped = ttnn.clip(logp_diff, -20.0, 20.0)
        ratio = ttnn.exp(logp_clipped)

        # surr1 = -adv * ratio, surr2 = -adv * clip(ratio, 1-eps, 1+eps)
        neg_adv = ttnn.multiply(adv, -1.0)
        surr1 = ttnn.multiply(neg_adv, ratio)
        clipped_ratio = ttnn.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        surr2 = ttnn.multiply(neg_adv, clipped_ratio)

        # max(surr1, surr2) - return per-sample, let caller do mean reduction
        surrogate = ttnn.maximum(surr1, surr2)

        # Save for backward
        ctx.diff = diff
        ctx.var = var
        ctx.ratio = ratio
        ctx.neg_adv = neg_adv
        ctx.clipped_ratio = clipped_ratio
        ctx.surr1 = surr1
        ctx.surr2 = surr2
        ctx.logp_diff = logp_diff
        ctx.A = A
        ctx.clip_eps = clip_eps

        return surrogate

    @staticmethod
    def backward(ctx, grad_output):
        # d(loss)/d(mean) via chain rule, all using raw ttnn ops.
        #
        # Which surrogate was active: surr1 >= surr2 means unclipped was used
        s1_wins = ttnn.ge(ctx.surr1, ctx.surr2)

        # Clip mask for ratio: 1 where ratio is within [1-eps, 1+eps]
        clip_mask = ttnn.multiply(
            ttnn.ge(ctx.ratio, 1.0 - ctx.clip_eps),
            ttnn.le(ctx.ratio, 1.0 + ctx.clip_eps)
        )

        # Effective d(loss)/d(ratio):
        # Where surr1 wins (unclipped): d = -adv
        # Where surr2 wins (clipped active): d = -adv * clip_mask
        # Combined: -adv * (s1_wins + (1-s1_wins) * clip_mask)
        # = -adv * (s1_wins + clip_mask - s1_wins * clip_mask)
        # Simpler: just use -adv * ratio_grad_mask
        # where ratio_grad_mask = s1_wins OR clip_mask
        ratio_grad_mask = ttnn.maximum(s1_wins, clip_mask)
        d_ratio = ttnn.multiply(ctx.neg_adv, ratio_grad_mask)

        # Logp clip mask
        logp_mask = ttnn.multiply(
            ttnn.ge(ctx.logp_diff, -20.0),
            ttnn.le(ctx.logp_diff, 20.0)
        )

        # d(ratio)/d(logp) = ratio; chain with d_ratio and logp mask
        d_logp = ttnn.multiply(ttnn.multiply(ctx.ratio, d_ratio), logp_mask)

        # d(logp)/d(mean) = diff / var * A
        d_mean_local = ttnn.multiply(
            ttnn.divide(ctx.diff, ctx.var),
            ctx.A
        )

        # Full gradient: d_logp * d_mean_local
        grad_mean = ttnn.multiply(d_logp, d_mean_local)

        # 5 tensor inputs: mean, actions, old_logp, adv, var - only mean needs grad
        return grad_mean, None, None, None, None


class Max(Function):
    @staticmethod
    def forward(ctx, a, b):
        a_val = a.get_value()
        b_val = b.get_value()
        result = ttnn.maximum(a_val, b_val)
        a_wins = ttnn.ge(a_val, b_val)
        ctx.a_wins = a_wins
        return result

    @staticmethod
    def backward(ctx, grad_output):
        a_wins = ctx.a_wins
        b_wins = ttnn.subtract(ttnn.ones_like(a_wins), a_wins)
        grad_a = ttnn.multiply(grad_output, a_wins)
        grad_b = ttnn.multiply(grad_output, b_wins)
        return grad_a, grad_b
