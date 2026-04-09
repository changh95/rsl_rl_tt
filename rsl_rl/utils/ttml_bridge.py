# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bridge utilities for torch <-> ttml tensor conversion.

TTML requires tile-aligned shapes (dimensions must be multiples of 32).
These helpers handle padding/unpadding transparently.
"""

from __future__ import annotations

import numpy as np
import torch

TILE_SIZE = 32

# Lazy imports to avoid importing ttml when not needed (e.g., on CUDA machines)
_ttml = None
_ttnn = None


def _ensure_ttml():
    """Lazily import ttml and ttnn."""
    global _ttml, _ttnn
    if _ttml is None:
        import ttml
        import ttnn
        _ttml = ttml
        _ttnn = ttnn
    return _ttml, _ttnn


def is_ttml_device(device: str) -> bool:
    """Check if the device string refers to a Tenstorrent NPU."""
    return isinstance(device, str) and device.startswith("ttml")


def pad_to_tile(dim: int) -> int:
    """Round up a dimension to the nearest multiple of TILE_SIZE."""
    return ((dim + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE


def torch_to_ttml(tensor: torch.Tensor, requires_grad: bool = False):
    """Convert a torch.Tensor [B, D] to a ttml Tensor [1, 1, B_pad, D_pad].

    Pads both batch and feature dims to multiples of 32 for tile alignment.
    For multi-device, shards the batch dimension across DDP devices.
    """
    ttml, ttnn = _ensure_ttml()

    arr = tensor.detach().cpu().numpy()
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    assert arr.ndim == 2, f"Expected 2D tensor, got {arr.ndim}D"

    B, D = arr.shape
    B_pad = pad_to_tile(B)
    D_pad = pad_to_tile(D)

    padded = np.zeros((B_pad, D_pad), dtype=np.float32)
    padded[:B, :D] = arr.astype(np.float32)
    # Shape [B_pad, 1, 1, D_pad] - batch first, matching ttml convention for DDP sharding on dim 0
    arr_4d = padded.reshape(B_pad, 1, 1, D_pad)

    # Check for multi-device - use mesh mapper to shard batch
    ctx = ttml.autograd.AutoContext.get_instance()
    device = ctx.get_device()
    num_devices = device.get_num_devices()

    if num_devices > 1:
        mapper = create_mesh_mapper(shard_batch=True)
        t = ttml.autograd.Tensor.from_numpy(arr_4d, ttnn.Layout.TILE, None, mapper)
    else:
        t = ttml.autograd.Tensor.from_numpy(arr_4d, layout=ttnn.Layout.TILE)

    if requires_grad:
        t.set_requires_grad(True)
    return t


def ttml_to_torch(ttml_tensor, original_shape: tuple[int, int]) -> torch.Tensor:
    """Convert a ttml Tensor [1, 1, B_pad, D_pad] back to torch.Tensor [B, D].

    Handles multi-device tensors by gathering with a mesh composer.
    """
    ttml, ttnn = _ensure_ttml()

    # Check if tensor is distributed across mesh
    ctx = ttml.autograd.AutoContext.get_instance()
    device = ctx.get_device()
    num_devices = device.get_num_devices()

    if num_devices > 1:
        # Gather distributed tensor to host using concat on batch dim
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)
        arr = ttml_tensor.to_numpy(composer=composer)
    else:
        arr = ttml_tensor.to_numpy(ttnn.DataType.FLOAT32)

    # Shape is [B_pad, 1, 1, D_pad] - squeeze middle dims
    if arr.ndim == 4:
        arr = arr.reshape(arr.shape[0], arr.shape[3])
    elif arr.ndim == 3:
        arr = arr.reshape(arr.shape[0], arr.shape[2])

    B, D = original_shape
    arr = arr[:B, :D]
    return torch.from_numpy(arr.copy())


def init_ttml_device(num_devices: int = 1):
    """Open the Tenstorrent device mesh.

    Args:
        num_devices: Number of devices to use.
            1 = single device (default, no fabric needed)
            4 = 4 devices in [2,2] mesh with DDP(2) + TP(2)

    Returns:
        (ctx, ddp_size) tuple.
    """
    import os
    ttml, ttnn = _ensure_ttml()
    ctx = ttml.autograd.AutoContext.get_instance()

    if num_devices <= 1:
        ctx.open_device()
        print("[ttml] Tenstorrent NPU device opened (single device).")
        return ctx, 1

    # Multi-device: enable fabric and open mesh
    os.environ.setdefault("TT_METAL_HOME", "/home/ttuser/tt-metal")
    os.environ.setdefault("TT_METAL_RUNTIME_ROOT", os.environ["TT_METAL_HOME"])
    ttml.core.distributed.enable_fabric(num_devices)

    if num_devices == 4:
        ctx.open_device([2, 2])
        ctx.initialize_parallelism_context(
            ttml.autograd.DistributedConfig(enable_ddp=True, enable_tp=True)
        )
    elif num_devices == 8:
        ctx.open_device([2, 4])
        ctx.initialize_parallelism_context(
            ttml.autograd.DistributedConfig(enable_ddp=True, enable_tp=True)
        )
    else:
        raise ValueError(f"Unsupported num_devices={num_devices}. Use 1, 4, or 8.")

    pctx = ctx.get_parallelism_context()
    ddp_size = pctx.get_ddp_size()
    tp_size = pctx.get_tp_size()
    print(f"[ttml] Tenstorrent NPU mesh opened: {num_devices} devices, DDP={ddp_size}, TP={tp_size}")
    return ctx, ddp_size


def sync_gradients(named_params):
    """Synchronize gradients across DDP devices."""
    ttml, _ = _ensure_ttml()
    ttml.core.distributed.synchronize_gradients(named_params)


def create_mesh_mapper(shard_batch: bool = True):
    """Create a mesh mapper for distributing data across devices.

    For DDP: shards batch dim across DDP axis, replicates across TP axis.
    Returns None if single-device (no sharding needed).
    """
    ttml, ttnn = _ensure_ttml()
    ctx = ttml.autograd.AutoContext.get_instance()
    device = ctx.get_device()
    num_devices = device.get_num_devices()

    if num_devices <= 1:
        return None

    shape = device.shape
    mesh_shape = ttnn.MeshShape(shape[0], shape[1])

    if shard_batch:
        config = ttnn.MeshMapperConfig(
            [ttnn.PlacementShard(0), ttnn.PlacementReplicate()],
            mesh_shape,
        )
    else:
        config = ttnn.MeshMapperConfig(
            [ttnn.PlacementReplicate(), ttnn.PlacementReplicate()],
            mesh_shape,
        )

    return ttnn.create_mesh_mapper(device, config)


def close_ttml_device():
    """Close the Tenstorrent device mesh."""
    ttml, _ = _ensure_ttml()
    ctx = ttml.autograd.AutoContext.get_instance()
    ctx.close_device()
    print("[ttml] Tenstorrent NPU device closed.")
