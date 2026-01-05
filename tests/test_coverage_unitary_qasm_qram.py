import math

import numpy as np
import pytest
from blockflow import BlockEncoding, ResourceEstimate, SuccessModel

from qargus.model import QuantumResNet, QuantumResNetConfig
from qargus.operators import DenseOperator
from qargus.qasm import (
    _collect_states,
    _decompose_state_preparation,
    _has_state_preparation,
    _normalize_state,
    _normalize_state_preparations,
    _pad_state,
    model_to_qasm,
)
from qargus.qram import qram_load_ry, qram_load_values
from qargus.regime3_unitary import _data_qubits_from_dim, _pooling_matrix
from qargus.unitary import (
    _blockflow_circuit_to_qiskit,
    _cached_erf_coeffs,
    _pad_matrix_to_power_of_two,
    _sqrtm_psd,
    block_encoding_circuit,
    block_encoding_circuit_from_block,
    block_encoding_unitary,
    chebyshev_to_monomial,
    concat_vector_encodings,
    erf_polynomial_coeffs,
    l2_pool_values,
    lcu_two_unitaries,
    matrix_vector_product_circuit,
    operator_to_matrix,
    square_polynomial_coeffs,
    unitary_dilation,
)


def test_pad_matrix_to_power_of_two():
    with pytest.raises(ValueError):
        _pad_matrix_to_power_of_two(np.ones(3))
    mat = np.arange(6, dtype=float).reshape(2, 3)
    padded = _pad_matrix_to_power_of_two(mat)
    assert padded.shape == (4, 4)
    assert np.allclose(padded[:2, :3], mat)
    square = np.eye(4)
    same = _pad_matrix_to_power_of_two(square)
    assert same is square


def test_sqrtm_psd_validation():
    with pytest.raises(ValueError):
        _sqrtm_psd(np.ones((2, 3)))
    mat = np.diag([4.0, 9.0])
    root = _sqrtm_psd(mat)
    assert np.allclose(root, np.diag([2.0, 3.0]))


def test_unitary_dilation_validation():
    with pytest.raises(ValueError):
        unitary_dilation(np.ones((2, 3)), alpha=1.0)
    with pytest.raises(ValueError):
        unitary_dilation(np.eye(2), alpha=0.0)


def test_block_encoding_unitary_validation():
    with pytest.raises(ValueError):
        block_encoding_unitary(np.eye(2), alpha=0.0)
    unitary, alpha = block_encoding_unitary(np.eye(2))
    assert unitary.shape == (4, 4)
    assert alpha > 0.0


def test_operator_to_matrix_validation():
    class DummyOp:
        shape = (2, 2)

        @staticmethod
        def apply(vec):
            return 2.0 * vec

    mat = operator_to_matrix(DummyOp())
    assert np.allclose(mat, 2.0 * np.eye(2))
    with pytest.raises(ValueError):
        operator_to_matrix(object())


def test_erf_coeff_helpers(monkeypatch):
    if not hasattr(np, "erf"):
        monkeypatch.setattr(np, "erf", np.vectorize(math.erf), raising=False)
    coeffs_np = _cached_erf_coeffs(3, (-1.0, 1.0), 1.0, hasattr(np, "erf"))
    coeffs_math = _cached_erf_coeffs(3, (-1.0, 1.0), 1.0, False)
    assert coeffs_np.shape[0] == 4
    assert coeffs_math.shape[0] == 4
    mono = chebyshev_to_monomial(coeffs_np)
    assert mono.shape[0] == coeffs_np.shape[0]
    coeffs = erf_polynomial_coeffs(degree=2)
    assert coeffs.shape[0] == 3
    assert np.allclose(square_polynomial_coeffs(), np.array([0.0, 0.0, 1.0]))


def test_l2_pool_values_validation():
    with pytest.raises(ValueError):
        l2_pool_values([1.0, 2.0, 3.0], pool=0)
    with pytest.raises(ValueError):
        l2_pool_values([1.0, 2.0, 3.0], pool=2)


def test_blockflow_circuit_to_qiskit_paths():
    pytest.importorskip("qiskit")

    class DummyGate:
        def __init__(self, name, qubits, controls=None, params=None):
            self.name = name
            self.qubits = qubits
            self.controls = controls or []
            self.params = params or []

    class DummyCircuit:
        def __init__(self, num_qubits, gates, name="dummy"):
            self.num_qubits = num_qubits
            self.gates = gates
            self.name = name

    gates = [
        DummyGate("x", [0]),
        DummyGate("y", [1]),
        DummyGate("z", [0]),
        DummyGate("h", [0]),
        DummyGate("s", [0]),
        DummyGate("t", [0]),
        DummyGate("rx", [0], params=[0.1]),
        DummyGate("ry", [0], params=[0.2]),
        DummyGate("rz", [0], params=[0.3]),
        DummyGate("cx", [0, 1]),
        DummyGate("cz", [0, 1]),
        DummyGate("swap", [0, 1]),
        DummyGate("x", [1], controls=[0]),
    ]
    qc = _blockflow_circuit_to_qiskit(DummyCircuit(2, gates), name="dummy")
    assert qc.num_qubits == 2

    with pytest.raises(ValueError):
        _blockflow_circuit_to_qiskit(DummyCircuit(1, [DummyGate("bad", [0])]))


def test_block_encoding_circuit_paths():
    pytest.importorskip("qiskit")
    qc = block_encoding_circuit(np.eye(2), alpha=0.5, data_first=False)
    assert qc.num_qubits == 2

    class DummyGate:
        def __init__(self, name, qubits, controls=None, params=None):
            self.name = name
            self.qubits = qubits
            self.controls = controls or []
            self.params = params or []

    class DummyCircuit:
        def __init__(self, num_qubits, gates):
            self.num_qubits = num_qubits
            self.gates = gates
            self.name = "dummy"

    class DummyBlock:
        def can_export_circuit(self):
            return True

        def build_circuit(self):
            return DummyCircuit(1, [DummyGate("x", [0])])

    qc = block_encoding_circuit_from_block(DummyBlock(), name="dummy", prefer_qublock=True)
    assert qc.num_qubits == 1

    block = type("Block", (), {"op": DenseOperator(np.eye(2)), "alpha": 1.0})()
    qc = block_encoding_circuit_from_block(block, prefer_qublock=False)
    assert qc.num_qubits == 2


def test_lcu_and_concat_validation():
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit

    u0 = QuantumCircuit(1)
    u1 = QuantumCircuit(1)
    with pytest.raises(ValueError):
        lcu_two_unitaries(u0, u1, coeffs=(-0.1, 0.2))
    with pytest.raises(ValueError):
        lcu_two_unitaries(u0, u1, coeffs=(0.0, 0.0))

    with pytest.raises(ValueError):
        concat_vector_encodings([])
    with pytest.raises(ValueError):
        concat_vector_encodings([QuantumCircuit(1), QuantumCircuit(2)])

    vector_prep = QuantumCircuit(2)
    block = QuantumCircuit(1)
    with pytest.raises(ValueError):
        matrix_vector_product_circuit(vector_prep, block)


def test_qasm_helper_validation():
    with pytest.raises(ValueError):
        _pad_state(np.ones((2, 2)), 4)
    with pytest.raises(ValueError):
        _pad_state(np.ones(3), 2)

    with pytest.raises(ValueError):
        _normalize_state(np.ones((2, 2)))
    with pytest.raises(ValueError):
        _normalize_state(np.array([1.0, np.nan]))
    with pytest.raises(ValueError):
        _normalize_state(np.zeros(2))

    assert np.allclose(_normalize_state(np.array([1.0, 0.0])), np.array([1.0, 0.0]))
    scaled = _normalize_state(np.array([2.0, 2.0]))
    assert np.allclose(scaled, np.array([1.0, 1.0]) / math.sqrt(2.0))

    class DummyInst:
        def __init__(self, name):
            self.operation = type("Op", (), {"name": name})()

    assert _has_state_preparation(type("QC", (), {"data": [DummyInst("state_preparation")]})())
    assert not _has_state_preparation(type("QC", (), {"data": [DummyInst("x")]})())


def test_qasm_state_preparation_normalization():
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import StatePreparation

    qc = QuantumCircuit(1)
    qc.append(StatePreparation([1.0, 1.0], normalize=True), [0])
    normalized = _normalize_state_preparations(qc)
    params = np.asarray(normalized.data[0].operation.params).reshape(-1)
    assert math.isclose(float(np.linalg.norm(params)), 1.0)

    inverse_qc = QuantumCircuit(1)
    inverse_qc.append(StatePreparation([1.0, 0.0], normalize=True).inverse(), [0])
    normalized_inverse = _normalize_state_preparations(inverse_qc)
    assert normalized_inverse.data[0].operation._inverse

    decomposed = _decompose_state_preparation(qc)
    assert not _has_state_preparation(decomposed)


def test_collect_states_paths():
    config = QuantumResNetConfig(
        input_shape=(1, 2, 2),
        num_classes=2,
        num_blocks=1,
        channels=1,
        kernel_size=1,
        output_mode="classifier",
    )
    model = QuantumResNet(config)
    x = np.array([[[1.0, 0.0], [0.0, 0.0]]])
    states = _collect_states(model, x)
    assert len(states) == config.num_blocks + 2

    config = QuantumResNetConfig(
        input_shape=(1, 2, 2),
        num_classes=2,
        num_blocks=0,
        channels=1,
        kernel_size=1,
        output_mode="regime3",
    )
    model = QuantumResNet(config)
    states = _collect_states(model, x)
    assert len(states) == 2


def test_model_to_qasm_validation_and_state_strategy():
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")
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
    with pytest.raises(ValueError):
        model_to_qasm(model, input_state=np.zeros((2, 2)))
    with pytest.raises(ValueError):
        model_to_qasm(model, input_state=x, strategy="bad")
    qasm = model_to_qasm(
        model,
        input_state=x,
        strategy="state",
        per_layer=False,
        include_measurements=True,
        basis_gates=["u", "cx"],
    )
    assert "OPENQASM 2.0;" in qasm
    assert "measure" in qasm.lower()


def test_qram_validation():
    pytest.importorskip("qiskit")
    with pytest.raises(ValueError):
        qram_load_values(1, 1, [0])
    with pytest.raises(ValueError):
        qram_load_values(1, 1, [0, 2])
    with pytest.raises(ValueError):
        qram_load_ry(1, 0, [0.0])
    with pytest.raises(ValueError):
        qram_load_ry(1, 0, [0.0, 1.5])


def test_regime3_helpers_validation():
    with pytest.raises(ValueError):
        _data_qubits_from_dim(0)
    with pytest.raises(ValueError):
        _pooling_matrix(4, 0)
    with pytest.raises(ValueError):
        _pooling_matrix(5, 2)
    mat = _pooling_matrix(4, 2)
    assert mat.shape == (2, 4)
