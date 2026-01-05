import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*BlueprintCircuit.*:DeprecationWarning"
)
from qiskit.quantum_info import Operator

from qargus.unitary import (
    block_encoding_circuit,
    concat_vector_encodings,
    erf_activation_circuit,
    lcu_two_unitaries,
    l2_pool_values,
    matrix_vector_product_circuit,
    polynomial_rotation_circuit,
    square_activation_circuit,
    tensor_product_circuit,
    unitary_dilation,
    vector_sum_circuit,
)


def _is_unitary(mat: np.ndarray, tol: float = 1e-8) -> bool:
    ident = np.eye(mat.shape[0], dtype=complex)
    return np.allclose(mat.conj().T @ mat, ident, atol=tol)


def test_unitary_dilation_is_unitary():
    a = np.array([[0.5, 0.1], [0.2, 0.3]], dtype=float)
    u = unitary_dilation(a, alpha=1.0)
    assert _is_unitary(u)


def test_block_encoding_circuit_is_unitary():
    a = np.array([[1.0, 0.0], [0.0, -0.5]])
    qc = block_encoding_circuit(a, alpha=1.5)
    u = Operator(qc).data
    assert _is_unitary(u)


def test_lcu_two_unitaries_is_unitary():
    from qiskit.circuit.library import XGate

    u0 = XGate().to_matrix()
    u1 = np.eye(2)
    u0_gate = block_encoding_circuit(u0, alpha=1.0).to_gate()
    u1_gate = block_encoding_circuit(u1, alpha=1.0).to_gate()
    qc = lcu_two_unitaries(u0_gate, u1_gate, coeffs=(0.4, 0.6))
    u = Operator(qc).data
    assert _is_unitary(u)


def test_vector_sum_circuit_is_unitary():
    u0_gate = block_encoding_circuit(np.eye(2), alpha=1.0).to_gate()
    u1_gate = block_encoding_circuit(np.eye(2), alpha=1.0).to_gate()
    qc = vector_sum_circuit(u0_gate, u1_gate, coeffs=(0.3, 0.7))
    u = Operator(qc).data
    assert _is_unitary(u)


def test_polynomial_rotation_circuit_unitary():
    gate = polynomial_rotation_circuit(2, coeffs=[0.0, 1.0, 0.5])
    u = Operator(gate).data
    assert _is_unitary(u)


def test_tensor_product_circuit_unitary():
    left = block_encoding_circuit(np.eye(2), alpha=1.0)
    right = block_encoding_circuit(np.eye(2), alpha=1.0)
    qc = tensor_product_circuit(left, right)
    u = Operator(qc).data
    assert _is_unitary(u)


def test_concat_vector_encodings_unitary():
    left = block_encoding_circuit(np.eye(2), alpha=1.0)
    right = block_encoding_circuit(np.eye(2), alpha=1.0)
    qc = concat_vector_encodings([left, right])
    u = Operator(qc).data
    assert _is_unitary(u)


def test_matrix_vector_product_circuit_unitary():
    vector_prep = block_encoding_circuit(np.eye(2), alpha=1.0)
    block = block_encoding_circuit(np.eye(2), alpha=1.0)
    qc = matrix_vector_product_circuit(vector_prep, block)
    u = Operator(qc).data
    assert _is_unitary(u)


def test_erf_activation_circuit_unitary():
    gate = erf_activation_circuit(2, degree=3)
    u = Operator(gate).data
    assert _is_unitary(u)


def test_square_activation_circuit_unitary():
    gate = square_activation_circuit(2)
    u = Operator(gate).data
    assert _is_unitary(u)


def test_l2_pool_values():
    vec = np.array([1.0, 2.0, 3.0, 4.0])
    pooled = l2_pool_values(vec, pool=2)
    expected = np.array([np.sqrt(5.0), 5.0])
    assert np.allclose(pooled, expected)
