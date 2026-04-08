# SPDX-License-Identifier: BSD-3-Clause

"""MLP module implemented with ttml LinearLayer and ttml.ops.unary activations.

CRITICAL: Activations MUST use ttml.ops.unary (not ttnn + create_tensor)
to preserve the autograd graph for gradient backpropagation through all layers.
"""

from __future__ import annotations

import ttml
from ttml.modules import AbstractModuleBase, LinearLayer, ModuleList

from rsl_rl_ttml.utils.tensor_utils import pad_to_tile


def resolve_activation(name: str):
    """Return a callable ttml.ops.unary activation that preserves the autograd graph."""
    activations = {
        "relu": ttml.ops.unary.relu,
        "gelu": ttml.ops.unary.gelu,
        "silu": ttml.ops.unary.silu,
        "identity": lambda x: x,
    }
    name = name.lower()
    if name in activations:
        return activations[name]
    raise ValueError(
        f"Unsupported activation '{name}'. Available (graph-safe): {list(activations.keys())}. "
        f"Note: elu/tanh/sigmoid are NOT available in ttml.ops.unary (would break autograd graph)."
    )


class MLP(AbstractModuleBase):
    """Multi-Layer Perceptron using ttml LinearLayer.

    All dimensions are tile-aligned (multiples of 32).
    Activations use ttml.ops.unary to preserve the autograd graph.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int],
        activation: str = "relu",
    ) -> None:
        super().__init__()

        self.activation_fn = resolve_activation(activation)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_dim_padded = pad_to_tile(input_dim)
        self.output_dim_padded = pad_to_tile(output_dim)

        hidden_dims = [input_dim if d == -1 else d for d in hidden_dims]

        self.layers = ModuleList()
        dims = [self.input_dim_padded] + [pad_to_tile(d) for d in hidden_dims] + [self.output_dim_padded]
        self._num_hidden = len(hidden_dims)

        for i in range(len(dims) - 1):
            self.layers.append(LinearLayer(dims[i], dims[i + 1]))

    def forward(self, x):
        """Forward pass: x is a ttml tensor [1, 1, B_pad, input_dim_padded]."""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self._num_hidden:
                x = self.activation_fn(x)
        return x
