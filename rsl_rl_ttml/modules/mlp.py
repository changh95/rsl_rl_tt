# SPDX-License-Identifier: BSD-3-Clause

"""MLP module implemented with ttml LinearLayer and ttnn activations.

All internal dimensions are padded to multiples of 32 (tile alignment).
Input/output padding is handled by the MLPModel wrapper.
"""

from __future__ import annotations

import ttnn
import ttml
from ttml.modules import AbstractModuleBase, LinearLayer, ModuleList

from rsl_rl_ttml.utils.tensor_utils import pad_to_tile


def resolve_activation(name: str):
    """Return a callable ttnn activation function from a name string."""
    activations = {
        "relu": ttnn.relu,
        "elu": lambda x: ttnn.elu(x, alpha=1.0),
        "selu": ttnn.selu,
        "tanh": ttnn.tanh,
        "sigmoid": ttnn.sigmoid,
        "gelu": ttnn.gelu,
        "silu": ttnn.silu,
        "identity": lambda x: x,
    }
    name = name.lower()
    if name in activations:
        return activations[name]
    raise ValueError(f"Unsupported activation '{name}'. Available: {list(activations.keys())}")


class MLP(AbstractModuleBase):
    """Multi-Layer Perceptron using ttml LinearLayer.

    All dimensions are tile-aligned (multiples of 32).
    Input shape: [1, 1, B_pad, D_pad] (tile layout)
    Output shape: [1, 1, B_pad, out_pad]
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int],
        activation: str = "elu",
    ) -> None:
        super().__init__()

        self.activation_fn = resolve_activation(activation)

        # Pad all dimensions to tile alignment
        # Note: LinearLayer handles tile alignment internally for weights,
        # but we need to ensure input/output dims match what the layer expects
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_dim_padded = pad_to_tile(input_dim)
        self.output_dim_padded = pad_to_tile(output_dim)

        # Resolve -1 dims
        hidden_dims = [input_dim if d == -1 else d for d in hidden_dims]

        # Build layers with padded dimensions
        self.layers = ModuleList()
        dims = [self.input_dim_padded] + [pad_to_tile(d) for d in hidden_dims] + [self.output_dim_padded]
        self._num_hidden = len(hidden_dims)

        for i in range(len(dims) - 1):
            self.layers.append(LinearLayer(dims[i], dims[i + 1]))

    def forward(self, x):
        """Forward pass: x is a ttml tensor [1, 1, B_pad, input_dim_padded]."""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply activation to hidden layers only (not the output layer)
            if i < self._num_hidden:
                x = ttml.autograd.create_tensor(
                    self.activation_fn(x.get_value()), requires_grad=True
                )
        return x
