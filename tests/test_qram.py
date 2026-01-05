import numpy as np
from qiskit.quantum_info import Operator

from qargus.qram import qram_load_ry, qram_load_values


def _is_unitary(mat: np.ndarray, tol: float = 1e-8) -> bool:
    ident = np.eye(mat.shape[0], dtype=complex)
    return np.allclose(mat.conj().T @ mat, ident, atol=tol)


def test_qram_load_values_unitary():
    qc = qram_load_values(2, 2, [0, 1, 2, 3])
    u = Operator(qc).data
    assert _is_unitary(u)


def test_qram_load_ry_unitary():
    qc = qram_load_ry(1, 1, [0.0, 0.5])
    u = Operator(qc).data
    assert _is_unitary(u)
