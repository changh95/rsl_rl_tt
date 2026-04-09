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
    arr_4d = padded.reshape(1, 1, B_pad, D_pad)

    t = ttml.autograd.Tensor.from_numpy(arr_4d, layout=ttnn.Layout.TILE)
    if requires_grad:
        t.set_requires_grad(True)
    return t


def ttml_to_torch(ttml_tensor, original_shape: tuple[int, int]) -> torch.Tensor:
    """Convert a ttml Tensor [1, 1, B_pad, D_pad] back to torch.Tensor [B, D]."""
    ttml, ttnn = _ensure_ttml()

    arr = ttml_tensor.to_numpy(ttnn.DataType.FLOAT32)
    if arr.ndim == 4:
        arr = arr.reshape(arr.shape[2], arr.shape[3])
    elif arr.ndim == 3:
        arr = arr.reshape(arr.shape[1], arr.shape[2])

    B, D = original_shape
    arr = arr[:B, :D]
    return torch.from_numpy(arr.copy())


def init_ttml_device():
    """Open the Tenstorrent device mesh."""
    ttml, _ = _ensure_ttml()
    ctx = ttml.autograd.AutoContext.get_instance()
    ctx.open_device()
    print("[ttml] Tenstorrent NPU device opened.")
    return ctx


def close_ttml_device():
    """Close the Tenstorrent device mesh."""
    ttml, _ = _ensure_ttml()
    ctx = ttml.autograd.AutoContext.get_instance()
    ctx.close_device()
    print("[ttml] Tenstorrent NPU device closed.")
