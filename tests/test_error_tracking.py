import numpy as np

from qargus.model import QuantumResNet, QuantumResNetConfig


def test_error_tracking_accumulates_configured_epsilons():
    config = QuantumResNetConfig(
        input_shape=(1, 1, 2),
        num_classes=2,
        num_blocks=1,
        channels=1,
        kernel_size=1,
        activation="identity",
        activation_error=0.1,
        normalize_error=0.2,
        pool_error=0.3,
        square_error=0.4,
        output_mode="regime3",
    )
    model = QuantumResNet(config)
    x = np.array([[[1.0, 2.0]]])
    result = model.forward(x)
    expected = 0.1 + (0.2 * 3) + 0.3 + 0.4
    assert np.isclose(result.trace.epsilon, expected)
