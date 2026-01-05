from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from blockflow import BlockEncoding, ResourceEstimate, SuccessModel

from .blocks import ResidualBlock
from .encoding import (
    EncodedTensor,
    Trace,
    VectorOpResult,
    apply_block_encoding,
    encoded_from_array,
    elementwise_square,
    flatten,
    l2_normalize,
    l2_pool_1d,
    pad_1d,
)
from .operators import Convolution2DOperator, DenseOperator


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


@dataclass(frozen=True)
class QuantumResNetConfig:
    input_shape: Tuple[int, int, int]
    num_classes: int = 10
    num_blocks: int = 2
    channels: int = 4
    kernel_size: int = 3
    activation: str = "erf"
    activation_scale: float = 0.8
    activation_error: float = 0.0
    normalize_error: float = 0.0
    pool_size: Optional[int] = None
    pool_error: float = 0.0
    square_error: float = 0.0
    output_mode: str = "regime3"
    seed: int = 7


@dataclass
class ForwardResult:
    logits: np.ndarray
    probabilities: np.ndarray
    encoded: EncodedTensor
    trace: Trace


class QuantumResNet:
    """
    Toy Regime-3 style residual CNN using QuBlock semantics.
    """

    def __init__(self, config: QuantumResNetConfig) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)
        channels, height, width = config.input_shape
        if channels != config.channels:
            raise ValueError("config.channels must match input_shape channels")
        if config.output_mode not in ("regime3", "classifier"):
            raise ValueError("output_mode must be 'regime3' or 'classifier'")
        if config.activation_error < 0.0:
            raise ValueError("activation_error must be non-negative")
        if config.normalize_error < 0.0:
            raise ValueError("normalize_error must be non-negative")
        if config.pool_error < 0.0:
            raise ValueError("pool_error must be non-negative")
        if config.square_error < 0.0:
            raise ValueError("square_error must be non-negative")
        padding = config.kernel_size // 2
        self.blocks: List[ResidualBlock] = []
        for _ in range(config.num_blocks):
            filters = rng.normal(
                scale=0.1,
                size=(config.channels, config.channels, config.kernel_size, config.kernel_size),
            )
            conv_op = Convolution2DOperator(
                filters=filters,
                input_shape=config.input_shape,
                stride=1,
                padding=padding,
            )
            conv_block = BlockEncoding(
                op=conv_op,
                alpha=float(np.linalg.norm(filters)),
                resources=ResourceEstimate(),
                success=SuccessModel(),
            )
            block = ResidualBlock(
                conv_block=conv_block,
                input_shape=config.input_shape,
                activation=config.activation,
                activation_scale=config.activation_scale,
                activation_error=config.activation_error,
                skip_tau=0.5,
                normalize_error=config.normalize_error,
            )
            self.blocks.append(block)

        self.flatten_dim = config.channels * height * width
        self.output_mode = config.output_mode
        if self.output_mode == "classifier":
            dense_mat = rng.normal(scale=0.1, size=(config.num_classes, self.flatten_dim))
            square_dim = max(self.flatten_dim, config.num_classes)
            square_mat = np.zeros((square_dim, square_dim))
            square_mat[: config.num_classes, : self.flatten_dim] = dense_mat
            dense_block = BlockEncoding(
                op=DenseOperator(square_mat),
                alpha=float(np.linalg.norm(square_mat)),
                resources=ResourceEstimate(),
                success=SuccessModel(),
            )
            self.classifier = dense_block
            self.classifier_out_dim = config.num_classes
            self.output_padded_dim = self.flatten_dim
            self.output_pool_size = None
        else:
            if config.num_classes <= 0:
                raise ValueError("num_classes must be positive")
            if config.pool_size is None:
                self.output_padded_dim = ((self.flatten_dim + config.num_classes - 1) // config.num_classes) * config.num_classes
                self.output_pool_size = self.output_padded_dim // config.num_classes
            else:
                if config.pool_size <= 0:
                    raise ValueError("pool_size must be positive")
                self.output_padded_dim = ((self.flatten_dim + config.pool_size - 1) // config.pool_size) * config.pool_size
                output_dim = self.output_padded_dim // config.pool_size
                if output_dim != config.num_classes:
                    raise ValueError("pool_size does not produce num_classes outputs")
                self.output_pool_size = config.pool_size
            self.classifier = None
            self.classifier_out_dim = config.num_classes

    def forward(self, x: np.ndarray, *, trace: Optional[Trace] = None) -> ForwardResult:
        encoded = encoded_from_array(x)
        run_trace = trace if trace is not None else Trace()
        if encoded.encoding.epsilon:
            run_trace.include(resources=ResourceEstimate(), success_prob=1.0, epsilon=encoded.encoding.epsilon)
        current = l2_normalize(encoded, trace=run_trace, epsilon=self.config.normalize_error)
        for block in self.blocks:
            current = block.apply(current.encoded, trace=current.trace)
        flat = flatten(current.encoded)
        if self.output_mode == "classifier":
            classified = apply_block_encoding(
                flat,
                self.classifier,
                out_shape=(self.config.num_classes,),
                output_dim=self.classifier_out_dim,
                trace=current.trace,
            )
            logits = classified.encoded.encoding.semantic_state()
            probs = _softmax(logits)
            return ForwardResult(
                logits=logits,
                probabilities=probs,
                encoded=classified.encoded,
                trace=classified.trace,
            )

        normalized = l2_normalize(flat, trace=current.trace, epsilon=self.config.normalize_error)
        if self.output_padded_dim > normalized.encoded.dimension:
            padded = pad_1d(normalized.encoded, self.output_padded_dim, trace=normalized.trace)
        else:
            padded = VectorOpResult(encoded=normalized.encoded, trace=normalized.trace)
        pooled = l2_pool_1d(
            padded.encoded,
            pool=self.output_pool_size,
            trace=padded.trace,
            epsilon=self.config.pool_error,
        )
        squared = elementwise_square(pooled.encoded, trace=pooled.trace, epsilon=self.config.square_error)
        probs = squared.encoded.encoding.semantic_state()
        total = float(np.sum(probs))
        if total > 0.0:
            probs = probs / total
        logits = np.log(probs + 1e-12)
        return ForwardResult(
            logits=logits,
            probabilities=probs,
            encoded=squared.encoded,
            trace=squared.trace,
        )
