import numpy as np

from qargus.model import QuantumResNet, QuantumResNetConfig
from qargus.weights import (
    conv_filters_from_keras,
    dense_weights_from_keras,
    load_classifier_weights,
    load_conv_filters,
    load_keras_model,
    load_sklearn_classifier,
)


class DummyEstimator:
    def __init__(self, coef):
        self.coef_ = coef


class DummyConv2D:
    def __init__(self, name, kernel):
        self.name = name
        self._kernel = kernel

    def get_weights(self):
        return [self._kernel, np.zeros(self._kernel.shape[-1])]


class DummyDense:
    def __init__(self, name, kernel):
        self.name = name
        self._kernel = kernel

    def get_weights(self):
        return [self._kernel, np.zeros(self._kernel.shape[-1])]


class DummyModel:
    def __init__(self, layers):
        self.layers = layers


def test_load_conv_and_classifier_weights():
    config = QuantumResNetConfig(
        input_shape=(1, 3, 3),
        num_classes=2,
        num_blocks=1,
        channels=1,
        kernel_size=3,
        output_mode="classifier",
    )
    model = QuantumResNet(config)
    filters = np.arange(9, dtype=float).reshape(1, 1, 3, 3)
    load_conv_filters(model, [filters])
    assert np.allclose(model.blocks[0].conv_block.op.filters, filters)

    weights = np.arange(18, dtype=float).reshape(2, 9)
    load_classifier_weights(model, weights)
    mat = model.classifier.op.mat
    assert np.allclose(mat[:2, :9], weights)


def test_load_sklearn_classifier():
    config = QuantumResNetConfig(
        input_shape=(1, 2, 2),
        num_classes=1,
        num_blocks=0,
        channels=1,
        kernel_size=1,
        output_mode="classifier",
    )
    model = QuantumResNet(config)
    coef = np.array([1.0, 2.0, 3.0, 4.0])
    estimator = DummyEstimator(coef)
    load_sklearn_classifier(model, estimator)
    mat = model.classifier.op.mat
    assert np.allclose(mat[:1, :4], coef.reshape(1, -1))


def test_keras_conversions_and_load():
    config = QuantumResNetConfig(
        input_shape=(1, 2, 2),
        num_classes=1,
        num_blocks=1,
        channels=1,
        kernel_size=1,
        output_mode="classifier",
    )
    model = QuantumResNet(config)
    keras_kernel = np.array([1.0], dtype=float).reshape(1, 1, 1, 1)
    keras_dense = np.arange(4, dtype=float).reshape(4, 1)
    conv_layer = DummyConv2D("conv2d", keras_kernel)
    dense_layer = DummyDense("dense", keras_dense)
    keras_model = DummyModel([conv_layer, dense_layer])
    load_keras_model(model, keras_model)

    expected_filters = conv_filters_from_keras(keras_kernel)
    expected_dense = dense_weights_from_keras(keras_dense)
    assert np.allclose(model.blocks[0].conv_block.op.filters, expected_filters)
    assert np.allclose(model.classifier.op.mat[:1, :4], expected_dense)
