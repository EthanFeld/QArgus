from __future__ import annotations

import math
from typing import Optional

import numpy as np

from blockflow import ResourceEstimate

from .operators import Convolution2DOperator, DenseOperator


def _qubits_for_dim(dim: int) -> int:
    if dim <= 0:
        raise ValueError("dimension must be positive")
    return int(math.ceil(math.log2(dim)))


def _ancillas_for_block(rows: int, cols: int) -> int:
    row_qubits = _qubits_for_dim(rows)
    col_qubits = _qubits_for_dim(cols)
    return max(1, (2 * row_qubits) + col_qubits + 9)


def estimate_block_encoding_resources(
    rows: int,
    cols: int,
    *,
    nonzeros: Optional[int] = None,
) -> ResourceEstimate:
    if rows <= 0 or cols <= 0:
        raise ValueError("rows/cols must be positive")
    if nonzeros is None:
        nonzeros = rows * cols
    nonzeros = int(nonzeros)
    if nonzeros < 1:
        nonzeros = 1
    ancillas = _ancillas_for_block(rows, cols)
    return ResourceEstimate(
        ancilla_qubits_clean=ancillas,
        t_count=nonzeros,
    )


def estimate_dense_resources(op: DenseOperator | np.ndarray) -> ResourceEstimate:
    if isinstance(op, DenseOperator):
        mat = op.mat
    else:
        mat = np.asarray(op)
    if mat.ndim != 2:
        raise ValueError("dense operator must be 2D")
    rows, cols = mat.shape
    nonzeros = int(np.count_nonzero(mat))
    return estimate_block_encoding_resources(int(rows), int(cols), nonzeros=nonzeros)


def estimate_conv_resources(op: Convolution2DOperator) -> ResourceEstimate:
    rows = int(op.output_dim())
    cols = int(op.input_dim())
    out_channels, _, kernel_h, kernel_w = op.filters.shape
    _, out_h, out_w = op.output_shape()
    per_position = int(np.count_nonzero(op.filters))
    nonzeros = per_position * int(out_h * out_w)
    return estimate_block_encoding_resources(rows, cols, nonzeros=nonzeros)


def estimate_activation_resources(dim: int, *, degree: int) -> ResourceEstimate:
    if degree <= 0:
        raise ValueError("degree must be positive")
    qubits = _qubits_for_dim(dim)
    t_count = max(1, int(degree * qubits))
    return ResourceEstimate(
        ancilla_qubits_clean=max(1, qubits + 1),
        t_count=t_count,
    )
