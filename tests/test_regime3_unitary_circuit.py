import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*BlueprintCircuit.*:DeprecationWarning"
)
from qiskit.quantum_info import Operator

from qargus.model import QuantumResNet, QuantumResNetConfig
from qargus.regime3_unitary import build_regime3_unitary_circuit


def _is_unitary(mat: np.ndarray, tol: float = 1e-8) -> bool:
    ident = np.eye(mat.shape[0], dtype=complex)
    return np.allclose(mat.conj().T @ mat, ident, atol=tol)


def test_build_regime3_unitary_circuit_unitary():
    config = QuantumResNetConfig(
        input_shape=(1, 2, 2),
        num_classes=2,
        num_blocks=1,
        channels=1,
        kernel_size=1,
        output_mode="regime3",
    )
    model = QuantumResNet(config)
    qc = build_regime3_unitary_circuit(model, activation_degree=3)
    u = Operator(qc).data
    assert _is_unitary(u)
