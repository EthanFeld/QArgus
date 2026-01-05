from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


@dataclass
class QuantumCountingResult:
    num_items: int
    true_count: int
    estimated_count: int
    precision_bits: int
    oracle_queries: int
    theta: float
    theta_est: float


def estimate_count(
    marks: Iterable[bool],
    *,
    precision_bits: int = 6,
) -> QuantumCountingResult:
    """
    Quantum-counting style estimate using phase estimation on Grover.

    This is a semantic simulation: it uses the true count to produce the
    estimate and tracks oracle query complexity. The phase estimate is mapped
    to the principal range [0, 1/2] to account for the two Grover eigenphases.
    """
    marks_list = list(marks)
    if not marks_list:
        raise ValueError("marks must be non-empty")
    if precision_bits <= 0:
        raise ValueError("precision_bits must be positive")
    num_items = len(marks_list)
    true_count = int(np.sum(marks_list))
    frac = true_count / num_items
    theta = math.asin(math.sqrt(frac)) if frac > 0 else 0.0
    scale = 2 ** precision_bits
    k = int(round(scale * theta / math.pi))
    k = max(0, min(scale - 1, k))
    phi_est = k / scale
    if phi_est > 0.5:
        phi_est = 1.0 - phi_est
    theta_est = phi_est * math.pi
    est_count = int(round(num_items * (math.sin(theta_est) ** 2)))
    oracle_queries = scale - 1
    return QuantumCountingResult(
        num_items=num_items,
        true_count=true_count,
        estimated_count=est_count,
        precision_bits=precision_bits,
        oracle_queries=oracle_queries,
        theta=theta,
        theta_est=theta_est,
    )
