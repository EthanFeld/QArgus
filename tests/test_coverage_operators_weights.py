import math

import numpy as np
import pytest
from blockflow import BlockEncoding, ResourceEstimate, SuccessModel

from qargus.model import QuantumResNet, QuantumResNetConfig
from qargus.operators import Convolution2DOperator, DenseOperator
from qargus.weights import (
    conv_filters_from_keras,
    dense_weights_from_keras,
    load_classifier_weights,
    load_conv_filters,
    load_keras_model,
    sklearn_linear_weights,
)


def test_dense_operator_validation_and_norm():
    mat = np.array([[1.0, 2.0], [3.0, 4.0]])
    op = DenseOperator(mat)
    assert op.shape == (2, 2)
    assert op.dtype == mat.dtype
    vec = np.array([1.0, 0.0])
    assert np.allclose(op.apply(vec), np.array([1.0, 3.0]))
    assert np.allclose(op.apply_adjoint(vec), np.array([1.0, 2.0]))
    first = op.norm_bound()
    second = op.norm_bound()
    assert math.isclose(first, second)
    with pytest.raises(ValueError):
        DenseOperator(np.array([1.0, 2.0]))


def test_convolution2d_operator_validation():
    with pytest.raises(ValueError):
        Convolution2DOperator(filters=np.zeros((1, 1, 1)), input_shape=(1, 2, 2))
    with pytest.raises(ValueError):
        Convolution2DOperator(filters=np.zeros((1, 1, 1, 1)), input_shape=(1, 2))
    with pytest.raises(ValueError):
        Convolution2DOperator(filters=np.zeros((1, 1, 1, 1)), input_shape=(1, 2, 2), stride=0)
    with pytest.raises(ValueError):
        Convolution2DOperator(filters=np.zeros((1, 1, 1, 1)), input_shape=(1, 2, 2), padding=-1)
    with pytest.raises(ValueError):
        Convolution2DOperator(filters=np.zeros((1, 2, 1, 1)), input_shape=(1, 2, 2))
    with pytest.raises(ValueError):
        Convolution2DOperator(filters=np.zeros((1, 1, 1, 1)), input_shape=(1, 0, 2))
    with pytest.raises(ValueError):
        Convolution2DOperator(filters=np.zeros((1, 1, 2, 2)), input_shape=(1, 1, 1))


def test_convolution2d_operator_apply_paths():
    filt = np.ones((1, 1, 1, 1))
    op = Convolution2DOperator(filters=filt, input_shape=(1, 3, 3), stride=2, padding=1)
    with pytest.raises(ValueError):
        op.apply(np.ones(3))
    out = op.apply(np.ones(op.input_dim()))
    assert out.shape[0] == op.output_dim()
    assert op.output_shape() == (1, 3, 3)
    assert op.input_dim() == 9

    identity = Convolution2DOperator(filters=filt, input_shape=(1, 2, 2))
    out_vec = np.array([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError):
        identity.apply_adjoint(np.ones(3))
    adj = identity.apply_adjoint(out_vec)
    assert np.allclose(adj, out_vec)

    padded = Convolution2DOperator(filters=filt, input_shape=(1, 2, 2), padding=1)
    adj_padded = padded.apply_adjoint(np.ones(padded.output_dim()))
    assert adj_padded.shape[0] == padded.input_dim()
    assert np.allclose(adj_padded, np.ones(padded.input_dim()))
    first = padded.norm_bound()
    second = padded.norm_bound()
    assert math.isclose(first, second)


def test_weight_conversion_validation():
    with pytest.raises(ValueError):
        conv_filters_from_keras(np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        dense_weights_from_keras(np.array([1.0, 2.0]))


def test_sklearn_linear_weights_validation():
    class NoCoef:
        pass

    with pytest.raises(ValueError):
        sklearn_linear_weights(NoCoef())
    with pytest.raises(ValueError):
        sklearn_linear_weights(type("Estimator", (), {"coef_": np.zeros((2, 2, 2))})())


def test_load_conv_filters_validation():
    class DummyModel:
        pass

    with pytest.raises(ValueError):
        load_conv_filters(DummyModel(), [])

    config = QuantumResNetConfig(
        input_shape=(1, 2, 2),
        num_classes=1,
        num_blocks=1,
        channels=1,
        kernel_size=1,
        output_mode="classifier",
    )
    model = QuantumResNet(config)
    with pytest.raises(ValueError):
        load_conv_filters(model, [])

    class NoConvBlock:
        conv_block = None

    model.blocks = [NoConvBlock()]
    with pytest.raises(ValueError):
        load_conv_filters(model, [np.zeros((1, 1, 1, 1))])

    bad_block = BlockEncoding(
        op=DenseOperator(np.eye(4)),
        alpha=1.0,
        resources=ResourceEstimate(),
        success=SuccessModel(),
    )
    model.blocks = [type("Block", (), {"conv_block": bad_block})()]
    with pytest.raises(ValueError):
        load_conv_filters(model, [np.zeros((1, 1, 1, 1))])

    conv_op = Convolution2DOperator(filters=np.ones((1, 1, 1, 1)), input_shape=(1, 2, 2))
    conv_block = BlockEncoding(
        op=conv_op,
        alpha=1.0,
        resources=ResourceEstimate(),
        success=SuccessModel(),
    )
    model.blocks = [type("Block", (), {"conv_block": conv_block})()]
    with pytest.raises(ValueError):
        load_conv_filters(model, [np.zeros((1, 1, 1))])


def test_load_classifier_weights_validation():
    config = QuantumResNetConfig(
        input_shape=(1, 2, 2),
        num_classes=2,
        num_blocks=0,
        channels=1,
        kernel_size=1,
        output_mode="regime3",
    )
    model = QuantumResNet(config)
    with pytest.raises(ValueError):
        load_classifier_weights(model, np.zeros((2, 4)))

    config = QuantumResNetConfig(
        input_shape=(1, 2, 2),
        num_classes=2,
        num_blocks=0,
        channels=1,
        kernel_size=1,
        output_mode="classifier",
    )
    model = QuantumResNet(config)
    with pytest.raises(ValueError):
        load_classifier_weights(model, np.zeros(4))
    with pytest.raises(ValueError):
        load_classifier_weights(model, np.zeros((2, 5)))


def test_load_keras_model_validation():
    config = QuantumResNetConfig(
        input_shape=(1, 2, 2),
        num_classes=1,
        num_blocks=1,
        channels=1,
        kernel_size=1,
        output_mode="classifier",
    )
    model = QuantumResNet(config)

    class NoLayers:
        layers = None

    with pytest.raises(ValueError):
        load_keras_model(model, NoLayers())

    class DummyConv2D:
        def __init__(self, name, kernel):
            self.name = name
            self._kernel = kernel

        def get_weights(self):
            return [self._kernel]

    conv_kernel = np.ones((1, 1, 1, 1))
    with pytest.raises(ValueError):
        load_keras_model(
            QuantumResNet(
                QuantumResNetConfig(
                    input_shape=(1, 2, 2),
                    num_classes=1,
                    num_blocks=2,
                    channels=1,
                    kernel_size=1,
                    output_mode="classifier",
                )
            ),
            type("KerasModel", (), {"layers": [DummyConv2D("conv2d", conv_kernel)]})(),
        )

    class DummyDense:
        def __init__(self, name, kernel):
            self.name = name
            self._kernel = kernel

        def get_weights(self):
            return [self._kernel]

    keras_model = type(
        "KerasModel",
        (),
        {"layers": [DummyConv2D("conv2d", conv_kernel), DummyDense("dense", np.ones((4, 1)))]},
    )()
    with pytest.raises(ValueError):
        load_keras_model(model, keras_model, dense_layer_name="missing_dense")

    class DummyConv2DEmptyWeights:
        def __init__(self, name):
            self.name = name

        def get_weights(self):
            return []

    with pytest.raises(ValueError):
        load_keras_model(model, type("KerasModel", (), {"layers": [DummyConv2DEmptyWeights("conv2d")]})())

    class DummyConv2DNoKernel:
        name = "conv2d"

    with pytest.raises(ValueError):
        load_keras_model(model, type("KerasModel", (), {"layers": [DummyConv2DNoKernel()]})())
