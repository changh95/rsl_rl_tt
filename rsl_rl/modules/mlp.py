# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from functools import reduce

from rsl_rl.utils import get_param, resolve_nn_activation
from rsl_rl.utils.ttml_bridge import is_ttml_device, pad_to_tile, torch_to_ttml, ttml_to_torch


class MLP(nn.Sequential):
    """Multi-Layer Perceptron.

    The MLP network is a sequence of linear layers and activation functions. The last layer is a linear layer that
    outputs the desired dimension unless the last activation function is specified.

    It provides additional conveniences:
    - If the hidden dimensions have a value of ``-1``, the dimension is inferred from the input dimension.
    - If the output dimension is a tuple, the output is reshaped to the desired shape.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int | tuple[int, ...] | list[int],
        hidden_dims: tuple[int, ...] | list[int],
        activation: str = "elu",
        last_activation: str | None = None,
    ) -> None:
        """Initialize the MLP.

        Args:
            input_dim: Dimension of the input.
            output_dim: Dimension of the output.
            hidden_dims: Dimensions of the hidden layers. A value of ``-1`` indicates that the dimension should be
                inferred from the input dimension.
            activation: Activation function.
            last_activation: Activation function of the last layer. None results in a linear last layer.
        """
        super().__init__()

        # Resolve activation functions
        activation_mod = resolve_nn_activation(activation)
        last_activation_mod = resolve_nn_activation(last_activation) if last_activation is not None else None
        # Resolve number of hidden dims if they are -1
        hidden_dims_processed = [input_dim if dim == -1 else dim for dim in hidden_dims]

        # Create layers sequentially
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims_processed[0]))
        layers.append(activation_mod)

        for layer_index in range(len(hidden_dims_processed) - 1):
            layers.append(nn.Linear(hidden_dims_processed[layer_index], hidden_dims_processed[layer_index + 1]))
            layers.append(activation_mod)

        # Add last layer
        if isinstance(output_dim, int):
            layers.append(nn.Linear(hidden_dims_processed[-1], output_dim))
        else:
            # Compute the total output dimension
            total_out_dim = reduce(lambda x, y: x * y, output_dim)
            # Add a layer to reshape the output to the desired shape
            layers.append(nn.Linear(hidden_dims_processed[-1], total_out_dim))
            layers.append(nn.Unflatten(dim=-1, unflattened_size=output_dim))

        # Add last activation function if specified
        if last_activation_mod is not None:
            layers.append(last_activation_mod)

        # Register the layers
        for idx, layer in enumerate(layers):
            self.add_module(f"{idx}", layer)

    def init_weights(self, scales: float | tuple[float]) -> None:
        """Initialize the weights of the MLP.

        Args:
            scales: Scale factor for the weights.
        """
        for idx, module in enumerate(self):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=get_param(scales, idx))
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP."""
        for layer in self:
            x = layer(x)
        return x


def _resolve_ttml_activation(name: str):
    """Return a ttml.ops.unary activation function that preserves the autograd graph."""
    import ttml
    activations = {
        "elu": ttml.ops.unary.relu,  # ELU not available in ttml.ops.unary; use ReLU as fallback
        "relu": ttml.ops.unary.relu,
        "gelu": ttml.ops.unary.gelu,
        "swish": ttml.ops.unary.silu,
        "silu": ttml.ops.unary.silu,
    }
    name = name.lower()
    if name in activations:
        return activations[name]
    # Fallback to ReLU with a warning
    print(f"[ttml] Activation '{name}' not available in ttml.ops.unary, falling back to relu")
    return ttml.ops.unary.relu


class TtmlMLP(nn.Module):
    """Drop-in replacement for MLP that runs on Tenstorrent NPU via ttml.

    Exposes the same nn.Module API (forward, parameters, state_dict) but internally
    uses ttml LinearLayer + ttml.ops.unary activations. The forward() method accepts
    and returns torch.Tensors, converting at the boundary.

    For training, use ``ttml_forward()`` to stay in ttml tensor space so the autograd
    graph is preserved for ``ttml.ops.loss`` backward passes.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int | tuple[int, ...] | list[int],
        hidden_dims: tuple[int, ...] | list[int],
        activation: str = "elu",
        last_activation: str | None = None,
    ) -> None:
        super().__init__()
        import ttml
        from ttml.modules import LinearLayer, ModuleList

        self.activation_fn = _resolve_ttml_activation(activation)
        self.last_activation_fn = _resolve_ttml_activation(last_activation) if last_activation else None

        # Store original (unpadded) dimensions
        self.input_dim = input_dim
        if isinstance(output_dim, int):
            self.output_dim = output_dim
        else:
            self.output_dim = reduce(lambda x, y: x * y, output_dim)

        # Resolve -1 hidden dims
        hidden_dims_processed = [input_dim if dim == -1 else dim for dim in hidden_dims]

        # Build ttml layers with tile-padded dimensions
        self._ttml_layers = ModuleList()
        dims = [pad_to_tile(input_dim)] + [pad_to_tile(d) for d in hidden_dims_processed] + [pad_to_tile(self.output_dim)]
        self._num_hidden = len(hidden_dims_processed)

        for i in range(len(dims) - 1):
            self._ttml_layers.append(LinearLayer(dims[i], dims[i + 1]))

        # Register dummy torch parameters so state_dict/load_state_dict work.
        # These are synced from ttml weights after each optimizer step.
        self._dummy_params = nn.ParameterList()
        for i in range(len(dims) - 1):
            w = nn.Parameter(torch.zeros(dims[i + 1], dims[i]), requires_grad=False)
            b = nn.Parameter(torch.zeros(dims[i + 1]), requires_grad=False)
            self._dummy_params.append(w)
            self._dummy_params.append(b)

    def ttml_forward(self, x_ttml):
        """Forward pass staying in ttml tensor space (preserves autograd graph for training)."""
        for i, layer in enumerate(self._ttml_layers):
            x_ttml = layer(x_ttml)
            if i < self._num_hidden:
                x_ttml = self.activation_fn(x_ttml)
        if self.last_activation_fn is not None:
            x_ttml = self.last_activation_fn(x_ttml)
        return x_ttml

    def ttml_named_parameters(self):
        """Return ttml NamedParameters for the ttml optimizer."""
        import ttml
        params = ttml.NamedParameters()
        for name, param in self._ttml_layers.named_parameters():
            params[f"ttml_mlp.{name}"] = param.tensor
        return params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: torch.Tensor in, torch.Tensor out. Runs on NPU internally."""
        B = x.shape[0]
        D_out = self.output_dim

        x_ttml = torch_to_ttml(x.view(B, -1))
        out_ttml = self.ttml_forward(x_ttml)
        result = ttml_to_torch(out_ttml, original_shape=(B, D_out))

        # Reset ttml graph after inference forward (no backward needed)
        import ttml
        ttml.autograd.AutoContext.get_instance().reset_graph()

        return result.to(x.device)

    def init_weights(self, scales: float | tuple[float]) -> None:
        """Initialize weights (approximate orthogonal via Xavier scaled by gain)."""
        import numpy as np
        import ttml
        import ttnn

        for idx, layer in enumerate(self._ttml_layers):
            scale = scales[idx] if isinstance(scales, (tuple, list)) else scales
            k = np.sqrt(6.0 / (layer.in_features + layer.out_features)) * scale
            weight_init = ttml.init.uniform(-k, k)
            shape = (1, 1, layer.out_features, layer.in_features)
            new_weight = weight_init(shape)
            layer.weight.tensor.set_value(
                ttml.autograd.Tensor.from_numpy(
                    np.array(new_weight).astype(np.float32), layout=ttnn.Layout.TILE
                ).get_value()
            )
