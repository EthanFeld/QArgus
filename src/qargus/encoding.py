from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple

import numpy as np

from blockflow import BlockEncoding, ResourceEstimate, SuccessModel, VectorEncoding


def _shape_size(shape: Sequence[int]) -> int:
    size = 1
    for dim in shape:
        if dim <= 0:
            raise ValueError("shape dimensions must be positive")
        size *= int(dim)
    return int(size)


@dataclass(frozen=True)
class EncodedTensor:
    """
    VectorEncoding plus tensor shape metadata.
    """
    encoding: VectorEncoding
    shape: Tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.shape, tuple) or not self.shape:
            raise ValueError("shape must be a non-empty tuple")
        if _shape_size(self.shape) != self.encoding.dimension:
            raise ValueError("shape size does not match encoded dimension")

    @property
    def dimension(self) -> int:
        return self.encoding.dimension

    def semantic_state(self) -> np.ndarray:
        return self.encoding.semantic_state().reshape(self.shape)


@dataclass
class Trace:
    """
    Accumulates resource and success information across semantic steps.
    """
    resources: ResourceEstimate = ResourceEstimate()
    success_prob: float = 1.0
    epsilon: float = 0.0

    def include(self, *, resources: ResourceEstimate, success_prob: float, epsilon: float = 0.0) -> None:
        self.resources = self.resources.combine(resources)
        self.success_prob *= float(success_prob)
        self.epsilon += float(epsilon)


@dataclass
class VectorOpResult:
    encoded: EncodedTensor
    trace: Trace


def encoded_from_array(
    array: np.ndarray,
    *,
    resources: Optional[ResourceEstimate] = None,
    success: Optional[SuccessModel] = None,
    epsilon: float = 0.0,
) -> EncodedTensor:
    """
    Create an EncodedTensor from a real or complex numpy array.
    """
    arr = np.asarray(array)
    vec = arr.reshape(-1)
    res = resources or ResourceEstimate()
    succ = success or SuccessModel()
    encoding = VectorEncoding.from_vector(
        vec,
        resources=res,
        success=succ,
        epsilon=epsilon,
    )
    return EncodedTensor(encoding=encoding, shape=arr.shape)


def flatten(encoded: EncodedTensor) -> EncodedTensor:
    return EncodedTensor(encoding=encoded.encoding, shape=(encoded.dimension,))


def pad_1d(
    encoded: EncodedTensor,
    padded_dim: int,
    *,
    trace: Optional[Trace] = None,
    epsilon: float = 0.0,
) -> VectorOpResult:
    """
    Zero-pad a 1D encoded vector to the requested dimension.
    """
    vec = encoded.encoding.semantic_state()
    if vec.ndim != 1:
        raise ValueError("pad_1d expects a 1D vector")
    if padded_dim < vec.shape[0]:
        raise ValueError("padded_dim must be >= current dimension")
    if padded_dim == vec.shape[0]:
        run_trace = trace if trace is not None else Trace()
        if not math.isclose(epsilon, 0.0):
            run_trace.include(resources=ResourceEstimate(), success_prob=1.0, epsilon=epsilon)
        return VectorOpResult(encoded=encoded, trace=run_trace)
    padded = np.zeros(padded_dim, dtype=vec.dtype)
    padded[: vec.shape[0]] = vec
    encoding = VectorEncoding.from_vector(
        padded,
        resources=ResourceEstimate(),
        success=SuccessModel(),
        epsilon=encoded.encoding.epsilon + epsilon,
    )
    out = EncodedTensor(encoding=encoding, shape=(padded_dim,))
    return _with_trace(out, trace, resources=encoding.resources, success_prob=encoding.success.success_prob, epsilon=epsilon)


def _with_trace(
    encoded: EncodedTensor,
    trace: Optional[Trace],
    *,
    resources: ResourceEstimate,
    success_prob: float,
    epsilon: float,
) -> VectorOpResult:
    new_trace = trace if trace is not None else Trace()
    new_trace.include(resources=resources, success_prob=success_prob, epsilon=epsilon)
    return VectorOpResult(encoded=encoded, trace=new_trace)


def l2_normalize(
    encoded: EncodedTensor,
    *,
    trace: Optional[Trace] = None,
    epsilon: float = 0.0,
) -> VectorOpResult:
    """
    L2-normalize the encoded vector.
    """
    vec = encoded.encoding.semantic_state()
    norm = float(np.linalg.norm(vec))
    if math.isclose(norm, 0.0):
        raise ValueError("cannot normalize a zero vector")
    new_vec = vec / norm
    encoding = VectorEncoding.from_vector(
        new_vec,
        resources=ResourceEstimate(),
        success=SuccessModel(),
        epsilon=encoded.encoding.epsilon + epsilon,
    )
    out = EncodedTensor(encoding=encoding, shape=encoded.shape)
    return _with_trace(out, trace, resources=encoding.resources, success_prob=encoding.success.success_prob, epsilon=epsilon)


def elementwise_square(
    encoded: EncodedTensor,
    *,
    trace: Optional[Trace] = None,
    epsilon: float = 0.0,
) -> VectorOpResult:
    """
    Element-wise square of amplitudes (|x|^2). Returns a real vector.
    """
    vec = encoded.encoding.semantic_state()
    squared = np.abs(vec)
    squared *= squared
    encoding = VectorEncoding.from_vector(
        squared,
        resources=ResourceEstimate(),
        success=SuccessModel(),
        epsilon=encoded.encoding.epsilon + epsilon,
    )
    out = EncodedTensor(encoding=encoding, shape=encoded.shape)
    return _with_trace(out, trace, resources=encoding.resources, success_prob=encoding.success.success_prob, epsilon=epsilon)


def l2_pool_1d(
    encoded: EncodedTensor,
    *,
    pool: int,
    trace: Optional[Trace] = None,
    epsilon: float = 0.0,
) -> VectorOpResult:
    """
    L2-pool a 1D vector into contiguous blocks of size pool.
    """
    if pool <= 0:
        raise ValueError("pool size must be positive")
    vec = encoded.encoding.semantic_state()
    if vec.ndim != 1:
        raise ValueError("l2_pool_1d expects a 1D vector")
    if vec.shape[0] % pool != 0:
        raise ValueError("pool size must divide vector length")
    grouped = vec.reshape(-1, pool)
    pooled = np.linalg.norm(grouped, axis=1)
    encoding = VectorEncoding.from_vector(
        pooled,
        resources=ResourceEstimate(),
        success=SuccessModel(),
        epsilon=encoded.encoding.epsilon + epsilon,
    )
    out = EncodedTensor(encoding=encoding, shape=(pooled.shape[0],))
    return _with_trace(out, trace, resources=encoding.resources, success_prob=encoding.success.success_prob, epsilon=epsilon)


def vector_sum(
    left: EncodedTensor,
    right: EncodedTensor,
    *,
    tau: float = 0.5,
    trace: Optional[Trace] = None,
    epsilon: float = 0.0,
) -> VectorOpResult:
    """
    Weighted sum of normalized vectors: tau * left + (1 - tau) * right.
    """
    if left.dimension != right.dimension:
        raise ValueError("vector_sum requires matching dimensions")
    if not (0.0 <= tau <= 1.0):
        raise ValueError("tau must be in [0, 1]")
    left_vec = left.encoding.semantic_state()
    right_vec = right.encoding.semantic_state()
    summed = (tau * left_vec) + ((1.0 - tau) * right_vec)
    encoding = VectorEncoding.from_vector(
        summed,
        resources=ResourceEstimate(),
        success=SuccessModel(),
        epsilon=left.encoding.epsilon + right.encoding.epsilon + epsilon,
    )
    out = EncodedTensor(encoding=encoding, shape=left.shape)
    return _with_trace(out, trace, resources=encoding.resources, success_prob=encoding.success.success_prob, epsilon=epsilon)


def apply_block_encoding(
    encoded: EncodedTensor,
    block: BlockEncoding,
    *,
    out_shape: Tuple[int, ...],
    output_dim: Optional[int] = None,
    trace: Optional[Trace] = None,
) -> VectorOpResult:
    """
    Apply a BlockEncoding to a normalized statevector and wrap the result.
    """
    state = encoded.encoding.semantic_state()
    out_vec = block.semantic_apply(state)
    if output_dim is not None:
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if output_dim > out_vec.shape[0]:
            raise ValueError("output_dim exceeds block output dimension")
        out_vec = out_vec[:output_dim]
    encoding = VectorEncoding.from_vector(
        out_vec,
        resources=block.resources,
        success=block.success,
        epsilon=encoded.encoding.epsilon + block.epsilon,
    )
    out = EncodedTensor(encoding=encoding, shape=out_shape)
    return _with_trace(out, trace, resources=block.resources, success_prob=block.success.success_prob, epsilon=block.epsilon)

