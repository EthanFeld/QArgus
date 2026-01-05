from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np

from blockflow.primitives.linear_operator import LinearOperator


def _prod(shape: Tuple[int, ...]) -> int:
    size = 1
    for dim in shape:
        if dim <= 0:
            raise ValueError("shape dimensions must be positive")
        size *= int(dim)
    return int(size)


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
        if self._norm_bound is not None:
            return float(self._norm_bound)
        return float(np.linalg.norm(self.mat, ord=2))


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

    def __post_init__(self) -> None:
        filters = np.asarray(self.filters)
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
        object.__setattr__(self, "filters", filters)

    @property
    def shape(self) -> tuple[int, int]:
        out_channels, _, kernel_h, kernel_w = self.filters.shape
        in_channels, height, width = self.input_shape
        out_h = (height + 2 * self.padding - kernel_h) // self.stride + 1
        out_w = (width + 2 * self.padding - kernel_w) // self.stride + 1
        if out_h <= 0 or out_w <= 0:
            raise ValueError("invalid output shape from convolution")
        out_dim = out_channels * out_h * out_w
        in_dim = in_channels * height * width
        return (int(out_dim), int(in_dim))

    @property
    def dtype(self):
        return self.filters.dtype

    def apply(self, vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec)
        in_channels, height, width = self.input_shape
        if vec.ndim != 1 or vec.shape[0] != in_channels * height * width:
            raise ValueError("vector dimension does not match convolution input")
        x = vec.reshape((in_channels, height, width))
        x_padded = np.pad(
            x,
            ((0, 0), (self.padding, self.padding), (self.padding, self.padding)),
            mode="constant",
        )
        out_channels, _, kernel_h, kernel_w = self.filters.shape
        out_h = (height + 2 * self.padding - kernel_h) // self.stride + 1
        out_w = (width + 2 * self.padding - kernel_w) // self.stride + 1
        out_dtype = np.result_type(self.filters.dtype, vec.dtype)
        out = np.zeros((out_channels, out_h, out_w), dtype=out_dtype)
        for oc in range(out_channels):
            for oh in range(out_h):
                for ow in range(out_w):
                    h0 = oh * self.stride
                    w0 = ow * self.stride
                    patch = x_padded[:, h0:h0 + kernel_h, w0:w0 + kernel_w]
                    out[oc, oh, ow] = np.sum(patch * self.filters[oc])
        return out.reshape(-1)

    def apply_adjoint(self, vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec)
        out_channels, in_channels, kernel_h, kernel_w = self.filters.shape
        _, height, width = self.input_shape
        out_h = (height + 2 * self.padding - kernel_h) // self.stride + 1
        out_w = (width + 2 * self.padding - kernel_w) // self.stride + 1
        if vec.ndim != 1 or vec.shape[0] != out_channels * out_h * out_w:
            raise ValueError("vector dimension does not match convolution output")
        y = vec.reshape((out_channels, out_h, out_w))
        out_dtype = np.result_type(self.filters.dtype, vec.dtype)
        x_grad = np.zeros((in_channels, height + 2 * self.padding, width + 2 * self.padding), dtype=out_dtype)
        filt = np.conjugate(self.filters)
        for oc in range(out_channels):
            for oh in range(out_h):
                for ow in range(out_w):
                    h0 = oh * self.stride
                    w0 = ow * self.stride
                    x_grad[:, h0:h0 + kernel_h, w0:w0 + kernel_w] += y[oc, oh, ow] * filt[oc]
        if self.padding > 0:
            x_grad = x_grad[:, self.padding:-self.padding, self.padding:-self.padding]
        return x_grad.reshape(-1)

    def norm_bound(self) -> float:
        return float(np.linalg.norm(self.filters))

    def output_shape(self) -> Tuple[int, int, int]:
        out_channels, _, kernel_h, kernel_w = self.filters.shape
        _, height, width = self.input_shape
        out_h = (height + 2 * self.padding - kernel_h) // self.stride + 1
        out_w = (width + 2 * self.padding - kernel_w) // self.stride + 1
        return (int(out_channels), int(out_h), int(out_w))

    def input_dim(self) -> int:
        return _prod(self.input_shape)

    def output_dim(self) -> int:
        shape = self.output_shape()
        return _prod(shape)
