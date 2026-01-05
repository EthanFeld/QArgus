from __future__ import annotations

import math
from typing import Iterable, Optional, Tuple

import numpy as np

from .activations import chebyshev_approximation
from .operators import Convolution2DOperator, DenseOperator


def _pad_to_power_of_two(vec: np.ndarray) -> np.ndarray:
    dim = vec.shape[0]
    target = 1 << int(math.ceil(math.log2(dim)))
    if target == dim:
        return vec
    padded = np.zeros(target, dtype=vec.dtype)
    padded[:dim] = vec
    return padded


def _pad_matrix_to_power_of_two(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat)
    if mat.ndim != 2:
        raise ValueError("matrix must be 2D")
    dim = max(mat.shape)
    target = 1 << int(math.ceil(math.log2(dim)))
    if mat.shape == (target, target):
        return mat
    padded = np.zeros((target, target), dtype=mat.dtype)
    padded[: mat.shape[0], : mat.shape[1]] = mat
    return padded


def _sqrtm_psd(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("matrix must be square")
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, 0.0, None)
    root = vecs @ np.diag(np.sqrt(vals)) @ vecs.conj().T
    return root


def unitary_dilation(matrix: np.ndarray, *, alpha: float) -> np.ndarray:
    """
    Build a unitary dilation for a matrix with ||matrix|| <= alpha.
    """
    mat = np.asarray(matrix, dtype=complex)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("matrix must be square")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    scaled = mat / alpha
    left = np.eye(mat.shape[0], dtype=complex) - scaled @ scaled.conj().T
    right = np.eye(mat.shape[0], dtype=complex) - scaled.conj().T @ scaled
    b = _sqrtm_psd(left)
    c = _sqrtm_psd(right)
    top = np.concatenate([scaled, -b], axis=1)
    bottom = np.concatenate([c, scaled.conj().T], axis=1)
    return np.concatenate([top, bottom], axis=0)


def block_encoding_unitary(matrix: np.ndarray, *, alpha: float | None = None) -> Tuple[np.ndarray, float]:
    """
    Return a unitary block-encoding for matrix and the chosen alpha.
    """
    mat = _pad_matrix_to_power_of_two(matrix)
    chosen_alpha = float(np.linalg.norm(mat, ord=2)) if alpha is None else float(alpha)
    if chosen_alpha <= 0.0:
        raise ValueError("alpha must be positive")
    unitary = unitary_dilation(mat, alpha=chosen_alpha)
    return unitary, chosen_alpha


def operator_to_matrix(op: object) -> np.ndarray:
    """
    Explicitly materialize a linear operator into a dense matrix.
    """
    if hasattr(op, "shape"):
        out_dim, in_dim = op.shape
    else:
        raise ValueError("operator must expose shape")
    mat = np.zeros((out_dim, in_dim), dtype=complex)
    for idx in range(in_dim):
        vec = np.zeros(in_dim, dtype=complex)
        vec[idx] = 1.0
        mat[:, idx] = op.apply(vec)
    return mat


def conv_operator_to_matrix(op: Convolution2DOperator) -> np.ndarray:
    return operator_to_matrix(op)


def dense_operator_to_matrix(op: DenseOperator) -> np.ndarray:
    return operator_to_matrix(op)


def chebyshev_to_monomial(coeffs: np.ndarray) -> np.ndarray:
    return np.polynomial.chebyshev.cheb2poly(coeffs)


def erf_polynomial_coeffs(
    *,
    degree: int,
    domain: Tuple[float, float] = (-1.0, 1.0),
    scale: float = 1.0,
) -> np.ndarray:
    coeffs = chebyshev_approximation(
        np.vectorize(lambda x: math.erf(scale * x)),
        degree=degree,
        domain=domain,
    )
    return chebyshev_to_monomial(coeffs)


def square_polynomial_coeffs() -> np.ndarray:
    return np.array([0.0, 0.0, 1.0])


def _as_unitary_gate(matrix: np.ndarray, name: str):
    from qiskit.circuit import Gate
    from qiskit.circuit.library import UnitaryGate

    gate = UnitaryGate(matrix, label=name)
    if not isinstance(gate, Gate):
        raise ValueError("failed to build unitary gate")
    return gate


def block_encoding_circuit(
    matrix: np.ndarray,
    *,
    alpha: float | None = None,
    name: str = "BlockEncode",
    data_first: bool = True,
):
    """
    Build a Qiskit circuit that applies the unitary block-encoding of matrix.
    """
    from qiskit import QuantumCircuit

    unitary, chosen_alpha = block_encoding_unitary(matrix, alpha=alpha)
    dim = unitary.shape[0]
    num_qubits = int(math.log2(dim))
    qc = QuantumCircuit(num_qubits, name=f"{name}_alpha_{chosen_alpha:.3f}")
    if data_first:
        ancilla_idx = num_qubits - 1
        for idx in range(ancilla_idx, 0, -1):
            qc.swap(idx, idx - 1)
        qc.append(_as_unitary_gate(unitary, name), qc.qubits)
        for idx in range(ancilla_idx):
            qc.swap(idx, idx + 1)
    else:
        qc.append(_as_unitary_gate(unitary, name), qc.qubits)
    return qc


def _blockflow_circuit_to_qiskit(circ, *, name: Optional[str] = None):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import (
        CXGate,
        CZGate,
        HGate,
        RXGate,
        RYGate,
        RZGate,
        SGate,
        SwapGate,
        TGate,
        XGate,
        YGate,
        ZGate,
    )

    qc = QuantumCircuit(circ.num_qubits, name=name or getattr(circ, "name", "blockflow"))
    for gate in circ.gates:
        name_lower = gate.name.lower()
        targets = list(gate.qubits)
        controls = list(gate.controls)

        if name_lower == "x":
            base = XGate()
        elif name_lower == "y":
            base = YGate()
        elif name_lower == "z":
            base = ZGate()
        elif name_lower == "h":
            base = HGate()
        elif name_lower == "s":
            base = SGate()
        elif name_lower == "t":
            base = TGate()
        elif name_lower == "rx":
            base = RXGate(gate.params[0])
        elif name_lower == "ry":
            base = RYGate(gate.params[0])
        elif name_lower == "rz":
            base = RZGate(gate.params[0])
        elif name_lower == "cx" and not controls:
            qc.append(CXGate(), targets)
            continue
        elif name_lower == "cz" and not controls:
            qc.append(CZGate(), targets)
            continue
        elif name_lower == "swap" and not controls:
            qc.append(SwapGate(), targets)
            continue
        else:
            raise ValueError(f"Unsupported blockflow gate: {gate}")

        if controls:
            qc.append(base.control(len(controls)), controls + targets)
        else:
            qc.append(base, targets)
    return qc


def block_encoding_circuit_from_block(
    block: object,
    *,
    name: str = "BlockEncode",
    prefer_qublock: bool = True,
):
    """
    Build a Qiskit block-encoding circuit from a BlockEncoding object.
    """
    if prefer_qublock and hasattr(block, "can_export_circuit") and block.can_export_circuit():
        circ = block.build_circuit()
        return _blockflow_circuit_to_qiskit(circ, name=name)
    matrix = operator_to_matrix(getattr(block, "op"))
    alpha = float(getattr(block, "alpha"))
    return block_encoding_circuit(matrix, alpha=alpha, name=name, data_first=True)


def lcu_two_unitaries(
    u0,
    u1,
    *,
    coeffs: Tuple[float, float],
    name: str = "LCU2",
):
    """
    Build a 2-term LCU selection circuit using a single select qubit.
    """
    from qiskit import QuantumCircuit

    c0, c1 = coeffs
    if c0 < 0.0 or c1 < 0.0:
        raise ValueError("coeffs must be non-negative")
    if math.isclose(c0 + c1, 0.0):
        raise ValueError("coeffs must not both be zero")
    norm = math.sqrt(c0 + c1)
    p0 = math.sqrt(c0) / norm
    theta = 2 * math.acos(p0)

    n_qubits = u0.num_qubits
    qc = QuantumCircuit(1 + n_qubits, name=name)
    qc.ry(theta, 0)
    qc.append(u0.control(1), [0] + list(range(1, n_qubits + 1)))
    qc.x(0)
    qc.append(u1.control(1), [0] + list(range(1, n_qubits + 1)))
    qc.x(0)
    qc.ry(-theta, 0)
    return qc


def grover_amplify(oracle, state_prep, iterations: int, name: str = "GroverAmp"):
    """
    Build a Grover-style amplitude amplification circuit.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import GroverOperator

    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    grover = GroverOperator(oracle, state_prep=state_prep, name=name)
    qc = QuantumCircuit(state_prep.num_qubits)
    qc.append(state_prep, qc.qubits)
    for _ in range(iterations):
        qc.append(grover, qc.qubits)
    return qc


def polynomial_rotation_circuit(
    num_state_qubits: int,
    coeffs: Iterable[float],
    *,
    basis: str = "Y",
    name: str = "poly",
):
    from qiskit.circuit.library import PolynomialPauliRotations

    return PolynomialPauliRotations(
        num_state_qubits=num_state_qubits,
        coeffs=list(coeffs),
        basis=basis,
        name=name,
    )


def erf_activation_circuit(
    num_state_qubits: int,
    *,
    degree: int = 5,
    domain: Tuple[float, float] = (-1.0, 1.0),
    basis: str = "Y",
    scale: float = 1.0,
):
    coeffs = erf_polynomial_coeffs(degree=degree, domain=domain, scale=scale)
    return polynomial_rotation_circuit(
        num_state_qubits,
        coeffs=coeffs,
        basis=basis,
        name="erf",
    )


def square_activation_circuit(num_state_qubits: int, *, basis: str = "Y"):
    coeffs = square_polynomial_coeffs()
    return polynomial_rotation_circuit(
        num_state_qubits,
        coeffs=coeffs,
        basis=basis,
        name="square",
    )


def l2_pool_values(vec: np.ndarray, pool: int) -> np.ndarray:
    arr = np.asarray(vec).reshape(-1)
    if pool <= 0:
        raise ValueError("pool must be positive")
    if arr.shape[0] % pool != 0:
        raise ValueError("pool must divide vector length")
    grouped = arr.reshape(-1, pool)
    pooled = np.sqrt(np.sum(np.abs(grouped) ** 2, axis=1))
    return pooled


def vector_sum_circuit(u0, u1, *, coeffs: Tuple[float, float], name: str = "VectorSum"):
    """
    Lemma 1: weighted sum of two vector-encoding unitaries via LCU.
    """
    return lcu_two_unitaries(u0, u1, coeffs=coeffs, name=name)


def tensor_product_circuit(u_left, u_right, *, name: str = "TensorProduct"):
    """
    Lemma 3: tensor product of two vector-encoding unitaries.
    """
    from qiskit import QuantumCircuit

    total = u_left.num_qubits + u_right.num_qubits
    qc = QuantumCircuit(total, name=name)
    qc.append(u_left, list(range(u_left.num_qubits)))
    qc.append(u_right, list(range(u_left.num_qubits, total)))
    return qc


def matrix_vector_product_circuit(
    vector_prep,
    block_circuit,
    *,
    name: str = "MatVec",
):
    """
    Lemma 2: apply a block-encoding to a prepared vector state.
    """
    from qiskit import QuantumCircuit

    if block_circuit.num_qubits < vector_prep.num_qubits:
        raise ValueError("block_circuit must have at least as many qubits as vector_prep")
    qc = QuantumCircuit(block_circuit.num_qubits, name=name)
    qc.append(vector_prep, list(range(vector_prep.num_qubits)))
    qc.append(block_circuit.to_gate(), list(range(block_circuit.num_qubits)))
    return qc


def concat_vector_encodings(unitaries: Iterable, *, name: str = "Concat"):
    """
    Lemma 4: concatenate vector encodings with a selector register.
    """
    from qiskit import QuantumCircuit

    unitary_list = list(unitaries)
    if not unitary_list:
        raise ValueError("unitaries must be non-empty")
    base_qubits = unitary_list[0].num_qubits
    if any(u.num_qubits != base_qubits for u in unitary_list):
        raise ValueError("all unitary circuits must have the same number of qubits")
    count = len(unitary_list)
    index_qubits = int(math.ceil(math.log2(count)))
    total = index_qubits + base_qubits
    qc = QuantumCircuit(total, name=name)
    index = list(range(index_qubits))
    data = list(range(index_qubits, total))

    for idx, unitary in enumerate(unitary_list):
        bits = [(idx >> bit) & 1 for bit in range(index_qubits)]
        for bit, qb in enumerate(index):
            if bits[bit] == 0:
                qc.x(qb)
        controlled = unitary.to_gate().control(index_qubits)
        qc.append(controlled, index + data)
        for bit, qb in enumerate(index):
            if bits[bit] == 0:
                qc.x(qb)
    return qc
