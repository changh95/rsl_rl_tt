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
        self._dummy_params = nn.ParameterList()
        for i in range(len(dims) - 1):
            w = nn.Parameter(torch.zeros(dims[i + 1], dims[i]), requires_grad=False)
            b = nn.Parameter(torch.zeros(dims[i + 1]), requires_grad=False)
            self._dummy_params.append(w)
            self._dummy_params.append(b)

        # CPU inference cache: numpy weight copies for fast CPU-only forward.
        # Synced from NPU weights via sync_weights_to_cpu() after each optimizer step.
        self._cpu_weights: list[tuple] | None = None  # [(W, b), ...] per layer
        self._dims = dims

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

    def sync_weights_to_cpu(self):
        """Copy NPU weights to CPU numpy cache for fast inference.

        Call this after each optimizer step so the CPU inference path
        uses the latest weights. This avoids expensive NPU round-trips
        during the rollout phase.
        """
        import numpy as np
        import ttml
        import ttnn

        ctx = ttml.autograd.AutoContext.get_instance()
        device = ctx.get_device()
        num_devices = device.get_num_devices()

        self._cpu_weights = []
        for layer in self._ttml_layers:
            if num_devices > 1:
                composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)
                W = layer.weight.tensor.to_numpy(composer=composer).astype(np.float32)
            else:
                W = layer.weight.tensor.to_numpy(ttnn.DataType.FLOAT32)
            # W shape: [1, 1, out, in] or [N, 1, out, in] for multi-device
            # Take first replica (weights are identical across DDP replicas)
            if W.ndim == 4:
                W = W[0, 0]  # [out, in]
            elif W.ndim == 3:
                W = W[0]

            if layer.bias is not None:
                if num_devices > 1:
                    b = layer.bias.tensor.to_numpy(composer=composer).astype(np.float32)
                else:
                    b = layer.bias.tensor.to_numpy(ttnn.DataType.FLOAT32)
                if b.ndim == 4:
                    b = b[0, 0, 0]  # [out]
                elif b.ndim == 3:
                    b = b[0, 0]
                elif b.ndim == 2:
                    b = b[0]
            else:
                b = None

            self._cpu_weights.append((W, b))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: torch.Tensor in, torch.Tensor out.

        Uses CPU numpy cache when available (fast path for rollout inference).
        Falls back to NPU forward when cache not synced.
        """
        import numpy as np

        B = x.shape[0]
        D_in = self.input_dim
        D_out = self.output_dim

        # Fast path: CPU inference using cached numpy weights
        if self._cpu_weights is not None:
            h = x.view(B, -1).detach().cpu().numpy().astype(np.float32)
            # Pad input to match padded dims
            D_pad = self._dims[0]
            if h.shape[1] < D_pad:
                h_padded = np.zeros((B, D_pad), dtype=np.float32)
                h_padded[:, :h.shape[1]] = h
                h = h_padded

            for i, (W, b) in enumerate(self._cpu_weights):
                h = h @ W.T  # [B, out_padded]
                if b is not None:
                    h = h + b
                # Activation for hidden layers
                if i < self._num_hidden:
                    h = np.maximum(h, 0)  # ReLU (matches ttml.ops.unary.relu)

            # Unpad output
            return torch.from_numpy(h[:B, :D_out].copy()).to(x.device)

        # Slow path: NPU forward (used before first sync)
        x_ttml = torch_to_ttml(x.view(B, -1))
        out_ttml = self.ttml_forward(x_ttml)
        result = ttml_to_torch(out_ttml, original_shape=(B, D_out))

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
