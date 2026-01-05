from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from blockflow import BlockEncoding, ResourceEstimate, SuccessModel, VectorEncoding

from .activations import activation_apply
from .encoding import EncodedTensor, Trace, VectorOpResult, apply_block_encoding, l2_normalize, vector_sum
from .resource_estimation import estimate_activation_resources


def _infer_conv_shape(block: BlockEncoding, default: Tuple[int, ...]) -> Tuple[int, ...]:
    op = block.op
    if hasattr(op, "output_shape"):
        return tuple(getattr(op, "output_shape")())
    return default


@dataclass
class ResidualBlock:
    conv_block: BlockEncoding
    input_shape: Tuple[int, int, int]
    activation: str = "erf"
    activation_scale: float = 0.8
    approx_coeffs: Optional[np.ndarray] = None
    activation_error: float = 0.0
    skip_tau: float = 0.5
    normalize_error: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.input_shape, tuple) or len(self.input_shape) != 3:
            raise ValueError("input_shape must be (channels, height, width)")
        if not (0.0 <= self.skip_tau <= 1.0):
            raise ValueError("skip_tau must be in [0, 1]")
        if self.activation_error < 0.0:
            raise ValueError("activation_error must be non-negative")
        if self.normalize_error < 0.0:
            raise ValueError("normalize_error must be non-negative")

    def apply(self, encoded: EncodedTensor, *, trace: Optional[Trace] = None) -> VectorOpResult:
        if encoded.shape != self.input_shape:
            raise ValueError("encoded shape does not match block input_shape")
        conv_shape = _infer_conv_shape(self.conv_block, encoded.shape)
        result = apply_block_encoding(encoded, self.conv_block, out_shape=conv_shape, trace=trace)
        act_vec = activation_apply(
            result.encoded.encoding.semantic_state(),
            kind=self.activation,
            approx_coeffs=self.approx_coeffs,
            scale=self.activation_scale,
        )
        act_encoding = VectorEncoding.from_vector(
            act_vec.reshape(-1),
            resources=self._activation_resources(result.encoded.dimension),
            success=SuccessModel(),
            epsilon=result.encoded.encoding.epsilon + self.activation_error,
        )
        act_encoded = EncodedTensor(encoding=act_encoding, shape=conv_shape)
        result.trace.include(
            resources=act_encoding.resources,
            success_prob=act_encoding.success.success_prob,
            epsilon=self.activation_error,
        )
        result = VectorOpResult(encoded=act_encoded, trace=result.trace)
        if result.encoded.dimension == encoded.dimension:
            result = vector_sum(encoded, result.encoded, tau=self.skip_tau, trace=result.trace)
        return l2_normalize(result.encoded, trace=result.trace, epsilon=self.normalize_error)

    def _activation_resources(self, dim: int) -> ResourceEstimate:
        if self.activation == "identity":
            return ResourceEstimate()
        if self.approx_coeffs is not None:
            degree = max(1, int(len(self.approx_coeffs) - 1))
        else:
            degree = 5
        return estimate_activation_resources(dim, degree=degree)
