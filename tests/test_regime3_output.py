import numpy as np

from qargus.encoding import (
    elementwise_square,
    encoded_from_array,
    flatten,
    l2_normalize,
    l2_pool_1d,
    pad_1d,
)
from qargus.model import QuantumResNet, QuantumResNetConfig


def test_regime3_output_matches_pool_and_square():
    config = QuantumResNetConfig(
        input_shape=(1, 1, 4),
        num_classes=2,
        num_blocks=0,
        channels=1,
        kernel_size=1,
        output_mode="regime3",
    )
    model = QuantumResNet(config)
    x = np.array([[[1.0, 2.0, 3.0, 4.0]]])
    result = model.forward(x)

    encoded = encoded_from_array(x)
    normalized = l2_normalize(encoded)
    flat = flatten(normalized.encoded)
    flat_norm = l2_normalize(flat, trace=normalized.trace)
    padded = pad_1d(flat_norm.encoded, 4, trace=flat_norm.trace)
    pooled = l2_pool_1d(padded.encoded, pool=2, trace=padded.trace)
    squared = elementwise_square(pooled.encoded, trace=pooled.trace)
    expected = squared.encoded.encoding.semantic_state()
    expected = expected / np.sum(expected)

    assert np.allclose(result.probabilities, expected)
    assert np.isclose(result.probabilities.sum(), 1.0)
