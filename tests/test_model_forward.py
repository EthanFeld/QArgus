import numpy as np

from qargus.model import QuantumResNet, QuantumResNetConfig


def test_model_forward_probabilities():
    config = QuantumResNetConfig(
        input_shape=(1, 4, 4),
        num_classes=3,
        num_blocks=1,
        channels=1,
        kernel_size=1,
        activation="identity",
    )
    model = QuantumResNet(config)
    x = np.random.default_rng(0).normal(size=config.input_shape)
    result = model.forward(x)
    assert result.probabilities.shape == (3,)
    assert abs(result.probabilities.sum() - 1.0) < 1e-6
