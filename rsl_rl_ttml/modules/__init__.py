# SPDX-License-Identifier: BSD-3-Clause

"""Modules package. Import specific modules directly to avoid eager ttnn imports."""

from rsl_rl_ttml.modules.normalization import EmpiricalNormalization
from rsl_rl_ttml.modules.distribution import GaussianDistribution

__all__ = ["EmpiricalNormalization", "GaussianDistribution"]

# MLP requires ttml/ttnn - import it explicitly when needed:
#   from rsl_rl_ttml.modules.mlp import MLP
