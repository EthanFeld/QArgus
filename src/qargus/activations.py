from __future__ import annotations

import math
from typing import Callable, Tuple

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _erf(x: np.ndarray) -> np.ndarray:
    vec_erf = np.vectorize(math.erf)
    return vec_erf(x)


def chebyshev_approximation(
    func: Callable[[np.ndarray], np.ndarray],
    *,
    degree: int,
    domain: Tuple[float, float] = (-1.0, 1.0),
    grid_size: int = 2048,
) -> np.ndarray:
    """
    Compute Chebyshev coefficients for func on the given domain.
    """
    if degree <= 0:
        raise ValueError("degree must be positive")
    a, b = domain
    if not a < b:
        raise ValueError("domain must satisfy a < b")
    x = np.linspace(a, b, grid_size)
    y = func(x)
    t = (2.0 * x - (b + a)) / (b - a)
    coeffs = np.polynomial.chebyshev.chebfit(t, y, degree)
    return coeffs


def _chebyshev_eval(x: np.ndarray, coeffs: np.ndarray, domain: Tuple[float, float]) -> np.ndarray:
    a, b = domain
    t = (2.0 * x - (b + a)) / (b - a)
    t = np.clip(t, -1.0, 1.0)
    return np.polynomial.chebyshev.chebval(t, coeffs)


def activation_apply(
    vec: np.ndarray,
    *,
    kind: str = "sigmoid",
    approx_coeffs: np.ndarray | None = None,
    approx_domain: Tuple[float, float] = (-1.0, 1.0),
    scale: float = 1.0,
) -> np.ndarray:
    """
    Apply an activation (sigmoid or erf) with optional Chebyshev approximation.
    """
    if kind not in ("sigmoid", "erf", "identity"):
        raise ValueError("kind must be sigmoid, erf, or identity")
    if kind == "identity":
        return vec
    if not math.isclose(scale, 1.0):
        vec = vec * scale
    if approx_coeffs is not None:
        return _chebyshev_eval(vec, approx_coeffs, approx_domain)
    if kind == "sigmoid":
        return _sigmoid(vec)
    return _erf(vec)
