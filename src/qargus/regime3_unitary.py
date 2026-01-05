from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np

from .unitary import (
    block_encoding_circuit,
    block_encoding_circuit_from_block,
    erf_activation_circuit,
    lcu_two_unitaries,
    square_activation_circuit,
)


def _data_qubits_from_dim(dim: int) -> int:
    if dim <= 0:
        raise ValueError("dimension must be positive")
    return int(math.ceil(math.log2(dim)))


def _as_gate(circuit, name: str):
    gate = circuit.to_gate()
    gate.name = name
    return gate


def _pooling_matrix(padded_dim: int, pool: int) -> np.ndarray:
    if pool <= 0:
        raise ValueError("pool must be positive")
    if padded_dim % pool != 0:
        raise ValueError("pool must divide padded_dim")
    out_dim = padded_dim // pool
    return np.kron(np.eye(out_dim, dtype=float), np.ones(pool, dtype=float))


def _max_block_dim(blocks: Iterable[object], fallback: int) -> int:
    dims = [fallback]
    for block in blocks:
        op = getattr(getattr(block, "conv_block", None), "op", None)
        if op is None or not hasattr(op, "shape"):
            continue
        dims.append(int(op.shape[0]))
    return max(dims)


def regime3_data_qubits(model: object) -> int:
    config = getattr(model, "config")
    input_dim = int(np.prod(config.input_shape))
    output_mode = getattr(model, "output_mode", "regime3")
    output_padded = int(getattr(model, "output_padded_dim", input_dim))
    max_dim = _max_block_dim(getattr(model, "blocks", []), input_dim)
    if output_mode == "regime3":
        max_dim = max(max_dim, output_padded)
    return _data_qubits_from_dim(max_dim)


def build_regime3_unitary_circuit(
    model: object,
    *,
    activation_degree: int = 5,
    include_output: bool = True,
    prefer_qublock: bool = True,
):
    """
    Build a Regime-3 unitary circuit using block-encoding + polynomial activation + LCU skip.

    This is a reference construction that uses QuBlock circuits when available and
    falls back to dense block-encoding otherwise.
    """
    from qiskit import QuantumCircuit

    config = getattr(model, "config")
    input_dim = int(np.prod(config.input_shape))
    output_mode = getattr(model, "output_mode", "regime3")
    output_padded = int(getattr(model, "output_padded_dim", input_dim))
    n_data = regime3_data_qubits(model)
    data_qubits = list(range(n_data))

    block_specs = []
    for idx, block in enumerate(getattr(model, "blocks", [])):
        conv_block = getattr(block, "conv_block")
        conv_circ = block_encoding_circuit_from_block(conv_block, name=f"Conv_{idx}", prefer_qublock=prefer_qublock)
        block_anc = conv_circ.num_qubits - _data_qubits_from_dim(int(conv_block.op.shape[0]))
        if block_anc < 0:
            raise ValueError("conv block circuit has fewer qubits than data register")
        block_specs.append((idx, block, conv_circ, block_anc))

    total_qubits = n_data
    for _, _, _, block_anc in block_specs:
        total_qubits += block_anc + 2

    output_spec = None
    if include_output:
        if output_mode == "classifier":
            classifier = getattr(model, "classifier", None)
            if classifier is None:
                raise ValueError("classifier output_mode requires model.classifier")
            cls_circ = block_encoding_circuit_from_block(classifier, name="Classifier", prefer_qublock=prefer_qublock)
            cls_anc = cls_circ.num_qubits - _data_qubits_from_dim(int(classifier.op.shape[0]))
            output_spec = ("classifier", cls_circ, cls_anc)
            total_qubits += max(cls_anc, 0)
        else:
            pool = int(getattr(model, "output_pool_size"))
            pool_mat = _pooling_matrix(output_padded, pool)
            pool_circ = block_encoding_circuit(pool_mat, name="Pool", data_first=True)
            pool_anc = pool_circ.num_qubits - _data_qubits_from_dim(output_padded)
            output_spec = ("pool", pool_circ, pool_anc)
            total_qubits += max(pool_anc, 0) + 1

    qc = QuantumCircuit(total_qubits)
    next_qubit = n_data

    for idx, block, conv_circ, block_anc in block_specs:
        ancillas = list(range(next_qubit, next_qubit + block_anc))
        act_qubit = next_qubit + block_anc
        select_qubit = next_qubit + block_anc + 1
        next_qubit += block_anc + 2

        conv_data_qubits = conv_circ.num_qubits - block_anc
        if conv_data_qubits > n_data:
            raise ValueError("conv block exceeds data register size")

        conv_gate = _as_gate(conv_circ, f"Conv_{idx}")
        if block.activation not in ("erf", "identity"):
            raise ValueError("unsupported activation for regime3 unitary")
        if block.activation == "identity":
            act_gate = _as_gate(QuantumCircuit(conv_data_qubits + 1), f"ActIdentity_{idx}")
        else:
            act_gate = _as_gate(
                erf_activation_circuit(conv_data_qubits, degree=activation_degree, scale=block.activation_scale),
                f"Erf_{idx}",
            )

        block_circ = QuantumCircuit(n_data + block_anc + 1)
        local_ancillas = list(range(n_data, n_data + block_anc))
        local_act_qubit = n_data + block_anc
        block_circ.append(conv_gate, data_qubits[:conv_data_qubits] + local_ancillas)
        block_circ.append(act_gate, data_qubits[:conv_data_qubits] + [local_act_qubit])
        block_gate = _as_gate(block_circ, f"Block_{idx}")

        identity = QuantumCircuit(n_data + block_anc + 1)
        identity_gate = _as_gate(identity, f"Identity_{idx}")
        coeffs = (block.skip_tau, 1.0 - block.skip_tau)
        lcu = lcu_two_unitaries(
            identity_gate,
            block_gate,
            coeffs=coeffs,
            name=f"Skip_{idx}",
        )

        lcu_qubits = [select_qubit] + data_qubits + ancillas + [act_qubit]
        qc.append(lcu.to_gate(), lcu_qubits)

    if include_output:
        if output_spec[0] == "classifier":
            _, cls_circ, cls_anc = output_spec
            ancillas = list(range(next_qubit, next_qubit + cls_anc))
            next_qubit += max(cls_anc, 0)
            cls_gate = _as_gate(cls_circ, "Classifier")
            cls_data_qubits = cls_circ.num_qubits - cls_anc
            if cls_data_qubits > n_data:
                raise ValueError("classifier block exceeds data register size")
            qc.append(cls_gate, data_qubits[:cls_data_qubits] + ancillas)
        else:
            _, pool_circ, pool_anc = output_spec
            square_qubit = next_qubit
            ancillas = list(range(next_qubit + 1, next_qubit + 1 + pool_anc))
            next_qubit += pool_anc + 1
            pool_data_qubits = pool_circ.num_qubits - pool_anc
            if pool_data_qubits > n_data:
                raise ValueError("pooling block exceeds data register size")
            square_gate = _as_gate(square_activation_circuit(pool_data_qubits), "Square")
            qc.append(square_gate, data_qubits[:pool_data_qubits] + [square_qubit])
            pool_gate = _as_gate(pool_circ, "Pool")
            if pool_anc:
                qc.append(pool_gate.control(1), [square_qubit] + data_qubits[:pool_data_qubits] + ancillas)
            else:
                qc.append(pool_gate.control(1), [square_qubit] + data_qubits[:pool_data_qubits])

    return qc
