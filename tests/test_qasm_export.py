import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*BlueprintCircuit.*:DeprecationWarning"
)

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")

from qargus.model import QuantumResNet, QuantumResNetConfig
from qargus.qasm import model_to_qasm


def test_model_to_qasm_classifier():
    config = QuantumResNetConfig(
        input_shape=(1, 2, 2),
        num_classes=2,
        num_blocks=0,
        channels=1,
        kernel_size=1,
        output_mode="classifier",
    )
    model = QuantumResNet(config)
    x = np.array([[[1.0, 0.0], [0.0, 0.0]]])
    qasm = model_to_qasm(model, input_state=x, include_measurements=False)
    assert "OPENQASM 2.0;" in qasm
    assert "qreg q[" in qasm
    assert "reset" not in qasm.lower()


def test_model_to_qasm_regime3():
    config = QuantumResNetConfig(
        input_shape=(1, 2, 2),
        num_classes=2,
        num_blocks=0,
        channels=1,
        kernel_size=1,
        output_mode="regime3",
    )
    model = QuantumResNet(config)
    x = np.array([[[1.0, 0.0], [0.0, 0.0]]])
    qasm = model_to_qasm(model, input_state=x, include_measurements=False)
    assert "OPENQASM 2.0;" in qasm
    assert "qreg q[" in qasm
