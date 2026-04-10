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
        arr = np.array(ttml_tensor.to_numpy(composer=composer), dtype=np.float32)
    else:
        # Use raw to_numpy() and cast - avoids slow on-device typecast
        arr = np.array(ttml_tensor.to_numpy(), dtype=np.float32)

    # Shape is [B_pad, 1, 1, D_pad] - squeeze middle dims
    if arr.ndim == 4:
        arr = arr.reshape(arr.shape[0], arr.shape[3])
    elif arr.ndim == 3:
        arr = arr.reshape(arr.shape[0], arr.shape[2])

    B, D = original_shape
    arr = arr[:B, :D]
    return torch.from_numpy(arr.copy())


def _detect_hardware():
    """Detect Tenstorrent hardware: architecture and device count."""
    _, ttnn = _ensure_ttml()
    arch = ttnn.get_arch_name()  # "wormhole_b0" or "blackhole"
    num_available = ttnn.GetNumAvailableDevices()
    num_pcie = ttnn.GetNumPCIeDevices()
    return arch, num_available, num_pcie


# Known mesh topologies per architecture and device count.
# Format: (mesh_shape, enable_ddp, enable_tp)
_MESH_CONFIGS = {
    # Wormhole B0 (T3K = 8 chips in 2x4, QuietBox = 1 chip)
    ("wormhole_b0", 1): (None, False, False),          # single device
    ("wormhole_b0", 4): ([2, 2], True, True),           # 4-chip: DDP=2, TP=2
    ("wormhole_b0", 8): ([2, 4], True, True),           # T3K: DDP=2, TP=4
    # Blackhole (single chip or multi-chip configurations)
    ("blackhole", 1):   (None, False, False),           # single device
    ("blackhole", 2):   ([1, 2], True, True),           # 2-chip
    ("blackhole", 4):   ([2, 2], True, True),           # 4-chip
    ("blackhole", 8):   ([2, 4], True, True),           # 8-chip (if available)
}


def init_ttml_device(num_devices: int = 0):
    """Open the Tenstorrent device mesh with auto-detection.

    Args:
        num_devices: Number of devices to use.
            0 = auto-detect (use all available devices)
            1 = single device (no fabric)
            N = use N devices with DDP

    Returns:
        (ctx, ddp_size) tuple.
    """
    import os
    ttml, ttnn = _ensure_ttml()
    ctx = ttml.autograd.AutoContext.get_instance()

    arch, num_available, num_pcie = _detect_hardware()

    if num_devices == 0:
        num_devices = num_available
    if num_devices > num_available:
        print(f"[ttml] Requested {num_devices} devices but only {num_available} available. Using {num_available}.")
        num_devices = num_available

    print(f"[ttml] Detected: arch={arch}, available={num_available}, pcie={num_pcie}, using={num_devices}")

    # Single device - no fabric needed
    if num_devices <= 1:
        ctx.open_device()
        print(f"[ttml] {arch} single device opened.")
        return ctx, 1

    # Multi-device: look up mesh config
    config_key = (arch, num_devices)
    if config_key not in _MESH_CONFIGS:
        # Fallback: try [1, N] with DDP only
        print(f"[ttml] No known mesh config for ({arch}, {num_devices}). Trying [1, {num_devices}].")
        mesh_shape = [1, num_devices]
        enable_ddp, enable_tp = True, True
    else:
        mesh_shape, enable_ddp, enable_tp = _MESH_CONFIGS[config_key]

    os.environ.setdefault("TT_METAL_HOME", os.environ.get("TT_METAL_HOME", "/home/ttuser/tt-metal"))
    os.environ.setdefault("TT_METAL_RUNTIME_ROOT", os.environ["TT_METAL_HOME"])

    ttml.core.distributed.enable_fabric(num_devices)
    ctx.open_device(mesh_shape)

    dc = ttml.autograd.DistributedConfig()
    dc.enable_ddp = enable_ddp
    dc.enable_tp = enable_tp
    ctx.initialize_parallelism_context(dc)

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
