from __future__ import annotations

import math
from typing import Iterable, List, Optional

import numpy as np

from .encoding import (
    apply_block_encoding,
    encoded_from_array,
    elementwise_square,
    flatten,
    l2_normalize,
    l2_pool_1d,
    pad_1d,
)
from .regime3_unitary import build_regime3_unitary_circuit, regime3_data_qubits


def _pad_state(vec: np.ndarray, target_dim: int) -> np.ndarray:
    if vec.ndim != 1:
        raise ValueError("state must be a 1D array")
    if target_dim < vec.shape[0]:
        raise ValueError("target_dim must be >= state dimension")
    padded = np.zeros(target_dim, dtype=vec.dtype)
    padded[: vec.shape[0]] = vec
    return padded


def _normalize_state(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec)
    if arr.ndim != 1:
        raise ValueError("state vector must be 1D for state preparation")
    finite_mask = np.isfinite(arr)
    if not np.all(finite_mask):
        bad = arr.size - int(np.sum(finite_mask))
        raise ValueError(f"state vector contains {bad} non-finite entries")
    max_abs = float(np.max(np.abs(arr)))
    if math.isclose(max_abs, 0.0):
        raise ValueError("cannot normalize a zero vector for state preparation")
    scaled = arr / max_abs
    norm = float(np.linalg.norm(scaled))
    if math.isclose(norm, 1.0):
        return scaled
    return scaled / norm


def _has_state_preparation(qc: object) -> bool:
    for inst in getattr(qc, "data", []):
        name = getattr(getattr(inst, "operation", None), "name", "")
        if name in ("state_preparation", "state_preparation_dg"):
            return True
    return False


def _normalize_state_preparations(qc: object) -> object:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import StatePreparation

    data = list(getattr(qc, "data", []))
    if not data:
        return qc

    new_qc = QuantumCircuit(*getattr(qc, "qregs", []), *getattr(qc, "cregs", []), name=qc.name)
    new_qc.global_phase = getattr(qc, "global_phase", 0.0)

    for inst in data:
        op = getattr(inst, "operation", None)
        name = getattr(op, "name", "")
        if name in ("state_preparation", "state_preparation_dg"):
            params = np.asarray(getattr(op, "params", []), dtype=np.complex128).reshape(-1)
            normalized = _normalize_state(params)
            new_op = StatePreparation(normalized, normalize=False)
            if name == "state_preparation_dg" or getattr(op, "_inverse", False):
                new_op = new_op.inverse()
            new_qc.append(new_op, inst.qubits, inst.clbits)
        else:
            new_qc.append(op, inst.qubits, inst.clbits)
    return new_qc


def _decompose_state_preparation(qc: object, max_passes: int = 3) -> object:
    if not _has_state_preparation(qc):
        return qc
    decomposed = _normalize_state_preparations(qc)
    for _ in range(max_passes):
        if not _has_state_preparation(decomposed):
            break
        decomposed = decomposed.decompose()
    return decomposed


def _collect_states(model: object, x: np.ndarray) -> List[np.ndarray]:
    config = getattr(model, "config")
    encoded = encoded_from_array(x)
    current = l2_normalize(encoded, epsilon=float(getattr(config, "normalize_error", 0.0)))
    states: List[np.ndarray] = [current.encoded.encoding.semantic_state().reshape(-1)]

    for block in getattr(model, "blocks", []):
        current = block.apply(current.encoded, trace=current.trace)
        states.append(current.encoded.encoding.semantic_state().reshape(-1))

    flat = flatten(current.encoded)
    output_mode = getattr(model, "output_mode", "regime3")
    if output_mode == "classifier":
        classified = apply_block_encoding(
            flat,
            getattr(model, "classifier"),
            out_shape=(int(getattr(config, "num_classes")),),
            output_dim=int(getattr(model, "classifier_out_dim")),
            trace=current.trace,
        )
        states.append(classified.encoded.encoding.semantic_state().reshape(-1))
        return states

    normalized = l2_normalize(flat, trace=current.trace, epsilon=float(getattr(config, "normalize_error", 0.0)))
    padded = pad_1d(
        normalized.encoded,
        int(getattr(model, "output_padded_dim")),
        trace=normalized.trace,
    )
    pooled = l2_pool_1d(
        padded.encoded,
        pool=int(getattr(model, "output_pool_size")),
        trace=padded.trace,
        epsilon=float(getattr(config, "pool_error", 0.0)),
    )
    squared = elementwise_square(
        pooled.encoded,
        trace=pooled.trace,
        epsilon=float(getattr(config, "square_error", 0.0)),
    )
    states.append(squared.encoded.encoding.semantic_state().reshape(-1))
    return states


def model_to_qasm(
    model: object,
    *,
    input_state: np.ndarray,
    include_measurements: bool = True,
    per_layer: bool = True,
    basis_gates: Optional[Iterable[str]] = None,
    optimization_level: int = 1,
    strategy: str = "unitary",
) -> str:
    """
    Export a QuantumResNet model as OpenQASM 2.0 using Qiskit/Aer.

    This exports a decomposed circuit for a specific input_state. The default
    strategy uses a unitary Regime-3 circuit; set strategy="state" to export the
    per-layer state preparation diagnostic.
    """
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import StatePreparation
    from qiskit.qasm2 import dumps as qasm2_dumps
    from qiskit_aer import Aer

    x = np.asarray(input_state)
    expected_shape = getattr(model, "config").input_shape
    if x.shape != expected_shape:
        raise ValueError("input_state shape must match model config.input_shape")

    if strategy not in ("unitary", "state"):
        raise ValueError("strategy must be 'unitary' or 'state'")

    if strategy == "state":
        states = _collect_states(model, x)
        max_dim = max(state.shape[0] for state in states)
        num_qubits = int(math.ceil(math.log2(max_dim)))
        target_dim = 2 ** num_qubits
        normalized_states = [_normalize_state(state) for state in states]
        padded_states = [_pad_state(state, target_dim) for state in normalized_states]

        qc = QuantumCircuit(num_qubits, num_qubits if include_measurements else 0)
        qc.append(StatePreparation(padded_states[0], normalize=True), qc.qubits)
        qc.barrier()

        if per_layer:
            prev = padded_states[0]
            for state in padded_states[1:]:
                qc.append(StatePreparation(prev, normalize=True).inverse(), qc.qubits)
                qc.append(StatePreparation(state, normalize=True), qc.qubits)
                qc.barrier()
                prev = state
    else:
        core = build_regime3_unitary_circuit(model)
        num_qubits = core.num_qubits
        target_dim = 2 ** int(regime3_data_qubits(model))
        state = encoded_from_array(x).encoding.semantic_state().reshape(-1)
        normalized_state = _normalize_state(state)
        padded_state = _pad_state(normalized_state, target_dim)
        qc = QuantumCircuit(num_qubits, num_qubits if include_measurements else 0)
        qc.append(
            StatePreparation(padded_state, normalize=True),
            list(range(int(regime3_data_qubits(model)))),
        )
        qc.append(core.to_gate(), qc.qubits)

    if include_measurements:
        qc.measure(qc.qubits, qc.clbits)

    qc = _decompose_state_preparation(qc)

    backend = Aer.get_backend("aer_simulator")
    if basis_gates is None:
        transpiled = transpile(qc, backend=backend, optimization_level=optimization_level)
    else:
        transpiled = transpile(
            qc,
            backend=backend,
            basis_gates=list(basis_gates),
            optimization_level=optimization_level,
        )
    return qasm2_dumps(transpiled)
