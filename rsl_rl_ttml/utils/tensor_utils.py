# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for converting between numpy arrays and ttml tensors.

TTML requires tile-aligned shapes: dimensions must be multiples of 32.
For 2D data [B, D], we convert to [1, 1, B, D_padded] where D_padded is
rounded up to the nearest multiple of 32. Batch dim B must also be a
multiple of 32 for TILE layout.

These helpers handle padding/unpadding transparently.
"""

from __future__ import annotations

import numpy as np

import ttnn
import ttml

TILE_SIZE = 32


def pad_to_tile(dim: int) -> int:
    """Round up a dimension to the nearest multiple of TILE_SIZE."""
    return ((dim + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE


def numpy_to_ttml(arr: np.ndarray, requires_grad: bool = False) -> ttml.autograd.Tensor:
    """Convert a numpy array [B, D] to a ttml tensor [1, 1, B_pad, D_pad].

    Pads both batch and feature dims to multiples of 32 for tile alignment.
    """
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    assert arr.ndim == 2, f"Expected 2D array, got {arr.ndim}D"

    B, D = arr.shape
    B_pad = pad_to_tile(B)
    D_pad = pad_to_tile(D)

    # Pad with zeros
    padded = np.zeros((B_pad, D_pad), dtype=np.float32)
    padded[:B, :D] = arr.astype(np.float32)

    # Reshape to [1, 1, B_pad, D_pad] for ttml TILE layout
    arr_4d = padded.reshape(1, 1, B_pad, D_pad)
    tensor = ttml.autograd.Tensor.from_numpy(arr_4d, layout=ttnn.Layout.TILE)
    if requires_grad:
        tensor.set_requires_grad(True)
    return tensor


def ttml_to_numpy(tensor: ttml.autograd.Tensor, original_shape: tuple[int, int] | None = None) -> np.ndarray:
    """Convert a ttml tensor [1, 1, B_pad, D_pad] back to numpy [B, D].

    Args:
        tensor: A ttml Tensor.
        original_shape: If provided, truncate padding to (B, D).
    """
    arr = tensor.to_numpy(ttnn.DataType.FLOAT32)
    # Handle various possible shapes from ttml
    if arr.ndim == 4:
        arr = arr.reshape(arr.shape[2], arr.shape[3])
    elif arr.ndim == 3:
        arr = arr.reshape(arr.shape[1], arr.shape[2])

    if original_shape is not None:
        B, D = original_shape
        arr = arr[:B, :D]

    return arr


def scalar_to_numpy(tensor: ttml.autograd.Tensor) -> float:
    """Extract a scalar value from a ttml tensor."""
    return float(tensor.to_numpy(ttnn.DataType.FLOAT32).flat[0])
