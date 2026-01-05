import math

import numpy as np

from qargus.activations import activation_apply


def test_activation_apply_scale_erf():
    vec = np.array([0.5, -0.5])
    scaled = activation_apply(vec, kind="erf", scale=2.0)
    expected = np.array([math.erf(1.0), math.erf(-1.0)])
    assert np.allclose(scaled, expected)
