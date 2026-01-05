from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from blockflow.primitives.linear_operator import LinearOperator


@dataclass(frozen=True)
class DenseOperator(LinearOperator):
    mat: np.ndarray
    _norm_bound: float | None = None

    def __post_init__(self) -> None:
        mat = np.asarray(self.mat)
        if mat.ndim != 2:
            raise ValueError("mat must be a 2D array")
        object.__setattr__(self, "mat", mat)

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.mat.shape[0]), int(self.mat.shape[1]))

    @property
    def dtype(self):
        return self.mat.dtype

    def apply(self, vec: np.ndarray) -> np.ndarray:
        return self.mat @ vec

    def apply_adjoint(self, vec: np.ndarray) -> np.ndarray:
        return self.mat.conj().T @ vec

    def norm_bound(self) -> float:
        if self._norm_bound is None:
            bound = float(np.linalg.norm(self.mat, ord=2))
            object.__setattr__(self, "_norm_bound", bound)
        return float(self._norm_bound)


@dataclass(frozen=True)
class Convolution2DOperator(LinearOperator):
    """
    Multi-filter 2D convolution as a linear operator.

    Input shape: (in_channels, height, width)
    Filters: (out_channels, in_channels, kernel_h, kernel_w)
    """
    filters: np.ndarray
    input_shape: Tuple[int, int, int]
    stride: int = 1
    padding: int = 0
    _output_shape: Tuple[int, int, int] = field(init=False, repr=False)
    _out_dim: int = field(init=False, repr=False)
    _in_dim: int = field(init=False, repr=False)
    _pad_width: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]] = field(init=False, repr=False)
    _norm_bound: float | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        filters = np.ascontiguousarray(self.filters)
        if filters.ndim != 4:
            raise ValueError("filters must be a 4D array")
        if len(self.input_shape) != 3:
            raise ValueError("input_shape must be (channels, height, width)")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if self.padding < 0:
            raise ValueError("padding must be non-negative")
        in_channels, height, width = self.input_shape
        if filters.shape[1] != in_channels:
            raise ValueError("filters in_channels must match input_shape")
        if height <= 0 or width <= 0:
            raise ValueError("input height/width must be positive")
        out_channels, _, kernel_h, kernel_w = filters.shape
        out_h = (height + 2 * self.padding - kernel_h) // self.stride + 1
        out_w = (width + 2 * self.padding - kernel_w) // self.stride + 1
        if out_h <= 0 or out_w <= 0:
            raise ValueError("invalid output shape from convolution")
        object.__setattr__(self, "filters", filters)
        object.__setattr__(self, "_output_shape", (int(out_channels), int(out_h), int(out_w)))
        object.__setattr__(self, "_in_dim", int(in_channels * height * width))
        object.__setattr__(self, "_out_dim", int(out_channels * out_h * out_w))
        object.__setattr__(
            self,
            "_pad_width",
            ((0, 0), (self.padding, self.padding), (self.padding, self.padding)),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self._out_dim), int(self._in_dim))

    @property
    def dtype(self):
        return self.filters.dtype

    def apply(self, vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec)
        in_channels, height, width = self.input_shape
        if vec.ndim != 1 or vec.shape[0] != self._in_dim:
            raise ValueError("vector dimension does not match convolution input")
        x = vec.reshape((in_channels, height, width))
        if self.padding:
            x = np.pad(
                x,
                self._pad_width,
                mode="constant",
            )
        out_channels, _, kernel_h, kernel_w = self.filters.shape
        _, out_h, out_w = self._output_shape
        windows = sliding_window_view(x, (kernel_h, kernel_w), axis=(1, 2))
        if self.stride > 1:
            windows = windows[:, ::self.stride, ::self.stride, :, :]
        out = np.tensordot(self.filters, windows, axes=([1, 2, 3], [0, 3, 4]))
        return out.reshape(-1)

    def apply_adjoint(self, vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec)
        out_channels, in_channels, kernel_h, kernel_w = self.filters.shape
        _, height, width = self.input_shape
        _, out_h, out_w = self._output_shape
        if vec.ndim != 1 or vec.shape[0] != self._out_dim:
            raise ValueError("vector dimension does not match convolution output")
        y = vec.reshape((out_channels, out_h, out_w))
        out_dtype = np.result_type(self.filters.dtype, vec.dtype)
        x_grad = np.zeros((in_channels, height + 2 * self.padding, width + 2 * self.padding), dtype=out_dtype)
        filt = np.conjugate(self.filters)
        stride = self.stride
        for kh in range(kernel_h):
            h_slice = slice(kh, kh + out_h * stride, stride)
            for kw in range(kernel_w):
                w_slice = slice(kw, kw + out_w * stride, stride)
                contrib = np.tensordot(filt[:, :, kh, kw], y, axes=([0], [0]))
                x_grad[:, h_slice, w_slice] += contrib
        if self.padding > 0:
            x_grad = x_grad[:, self.padding:-self.padding, self.padding:-self.padding]
        return x_grad.reshape(-1)

    def norm_bound(self) -> float:
        if self._norm_bound is None:
            bound = float(np.linalg.norm(self.filters))
            object.__setattr__(self, "_norm_bound", bound)
        return float(self._norm_bound)

    def output_shape(self) -> Tuple[int, int, int]:
        return self._output_shape

    def input_dim(self) -> int:
        return self._in_dim

    def output_dim(self) -> int:
        return self._out_dim
