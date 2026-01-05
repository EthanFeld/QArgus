from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def _bits(value: int, width: int) -> list[int]:
    return [(value >> idx) & 1 for idx in range(width)]


def qram_load_values(index_qubits: int, value_qubits: int, values: Sequence[int]):
    """
    Build a QRAM-style loader mapping |i>|0> -> |i>|values[i]>.
    """
    from qiskit import QuantumCircuit

    n = 1 << index_qubits
    if len(values) != n:
        raise ValueError("values length must be 2**index_qubits")
    max_val = (1 << value_qubits) - 1
    if any(val < 0 or val > max_val for val in values):
        raise ValueError("values must fit within value_qubits")

    qc = QuantumCircuit(index_qubits + value_qubits)
    for idx, val in enumerate(values):
        idx_bits = _bits(idx, index_qubits)
        for bit, qb in enumerate(idx_bits):
            if qb == 0:
                qc.x(bit)
        val_bits = _bits(val, value_qubits)
        for bit, vb in enumerate(val_bits):
            if vb == 1:
                qc.mcx(list(range(index_qubits)), index_qubits + bit)
        for bit, qb in enumerate(idx_bits):
            if qb == 0:
                qc.x(bit)
    return qc


def qram_load_ry(index_qubits: int, target_qubit: int, values: Sequence[float]):
    """
    Build a QRAM-style loader applying controlled RY rotations per index.
    """
    from qiskit import QuantumCircuit

    n = 1 << index_qubits
    if len(values) != n:
        raise ValueError("values length must be 2**index_qubits")
    if any(abs(val) > 1.0 for val in values):
        raise ValueError("values must be within [-1, 1]")

    qc = QuantumCircuit(index_qubits + 1)
    for idx, val in enumerate(values):
        idx_bits = _bits(idx, index_qubits)
        for bit, qb in enumerate(idx_bits):
            if qb == 0:
                qc.x(bit)
        theta = 2 * np.arcsin(float(val))
        qc.mcry(theta, list(range(index_qubits)), target_qubit)
        for bit, qb in enumerate(idx_bits):
            if qb == 0:
                qc.x(bit)
    return qc
