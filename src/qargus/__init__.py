"""Lean semantic Regime-3 prototype from Rattew et al. (arXiv:2510.07195)."""

from .encoding import (
    EncodedTensor,
    VectorOpResult,
    apply_block_encoding,
    elementwise_square,
    l2_normalize,
    l2_pool_1d,
    pad_1d,
    vector_sum,
    flatten,
)
from .operators import Convolution2DOperator, DenseOperator
from .activations import activation_apply, chebyshev_approximation
from .blocks import ResidualBlock
from .model import QuantumResNet, QuantumResNetConfig, ForwardResult
from .counting import QuantumCountingResult, estimate_count
from .qasm import model_to_qasm
from .unitary import (
    block_encoding_circuit,
    block_encoding_circuit_from_block,
    block_encoding_unitary,
    chebyshev_to_monomial,
    erf_polynomial_coeffs,
    erf_activation_circuit,
    l2_pool_values,
    lcu_two_unitaries,
    concat_vector_encodings,
    matrix_vector_product_circuit,
    polynomial_rotation_circuit,
    square_activation_circuit,
    square_polynomial_coeffs,
    tensor_product_circuit,
    unitary_dilation,
    vector_sum_circuit,
)
from .qram import qram_load_ry, qram_load_values
from .regime3_unitary import build_regime3_unitary_circuit, regime3_data_qubits
from .semantic import SemanticResult, build_semantic_program, run_semantic_model
from .weights import (
    conv_filters_from_keras,
    dense_weights_from_keras,
    load_classifier_weights,
    load_conv_filters,
    load_keras_model,
    load_sklearn_classifier,
    sklearn_linear_weights,
)

__all__ = [
    "EncodedTensor",
    "VectorOpResult",
    "apply_block_encoding",
    "elementwise_square",
    "l2_normalize",
    "l2_pool_1d",
    "pad_1d",
    "vector_sum",
    "flatten",
    "Convolution2DOperator",
    "DenseOperator",
    "activation_apply",
    "chebyshev_approximation",
    "ResidualBlock",
    "QuantumResNet",
    "QuantumResNetConfig",
    "ForwardResult",
    "QuantumCountingResult",
    "estimate_count",
    "model_to_qasm",
    "block_encoding_circuit",
    "block_encoding_circuit_from_block",
    "block_encoding_unitary",
    "chebyshev_to_monomial",
    "erf_polynomial_coeffs",
    "erf_activation_circuit",
    "l2_pool_values",
    "lcu_two_unitaries",
    "concat_vector_encodings",
    "matrix_vector_product_circuit",
    "polynomial_rotation_circuit",
    "square_activation_circuit",
    "square_polynomial_coeffs",
    "tensor_product_circuit",
    "unitary_dilation",
    "vector_sum_circuit",
    "qram_load_ry",
    "qram_load_values",
    "build_regime3_unitary_circuit",
    "regime3_data_qubits",
    "SemanticResult",
    "build_semantic_program",
    "run_semantic_model",
    "conv_filters_from_keras",
    "dense_weights_from_keras",
    "load_classifier_weights",
    "load_conv_filters",
    "load_keras_model",
    "load_sklearn_classifier",
    "sklearn_linear_weights",
]
