import numpy as np

from qargus.model import QuantumResNet, QuantumResNetConfig
from qargus.semantic import run_semantic_model


def test_semantic_classifier_matches_forward():
    config = QuantumResNetConfig(
        input_shape=(1, 2, 2),
        num_classes=2,
        num_blocks=1,
        channels=1,
        kernel_size=1,
        output_mode="classifier",
    )
    model = QuantumResNet(config)
    x = np.array([[[1.0, 0.0], [0.5, -0.5]]])
    semantic = run_semantic_model(model, x)
    forward = model.forward(x)
    assert np.allclose(semantic.state, forward.logits, atol=1e-8)
    assert semantic.report.uses == config.num_blocks + 1


def test_semantic_regime3_matches_forward():
    config = QuantumResNetConfig(
        input_shape=(1, 2, 2),
        num_classes=2,
        num_blocks=1,
        channels=1,
        kernel_size=1,
        output_mode="regime3",
    )
    model = QuantumResNet(config)
    x = np.array([[[0.25, -0.5], [0.75, 0.5]]])
    semantic = run_semantic_model(model, x)
    forward = model.forward(x)
    assert np.allclose(semantic.state, forward.probabilities, atol=1e-8)
    assert semantic.report.uses == config.num_blocks
