import math

import numpy as np
import pytest
from blockflow import BlockEncoding, ResourceEstimate, SuccessModel, VectorEncoding
from blockflow.semantic.state import StateVector
from blockflow.semantic.tracking import RunReport

from qargus.activations import activation_apply, chebyshev_approximation
from qargus.blocks import ResidualBlock
from qargus.counting import estimate_count
from qargus.encoding import (
    EncodedTensor,
    Trace,
    apply_block_encoding,
    encoded_from_array,
    l2_pool_1d,
    pad_1d,
    vector_sum,
)
from qargus.model import QuantumResNet, QuantumResNetConfig
from qargus.operators import DenseOperator
from qargus.resource_estimation import (
    _qubits_for_dim,
    estimate_activation_resources,
    estimate_block_encoding_resources,
    estimate_dense_resources,
)
from qargus.semantic import Regime3OutputStep, _l2_normalize, build_semantic_program, run_semantic_model


def test_chebyshev_approximation_validation():
    with pytest.raises(ValueError):
        chebyshev_approximation(lambda x: x, degree=0)
    with pytest.raises(ValueError):
        chebyshev_approximation(lambda x: x, degree=2, domain=(1.0, 0.0))


def test_activation_identity_and_approx_coeffs():
    vec = np.array([0.2, -0.2])
    assert np.allclose(activation_apply(vec, kind="identity"), vec)
    coeffs = np.array([0.5])
    approx = activation_apply(vec, kind="sigmoid", approx_coeffs=coeffs, approx_domain=(-1.0, 1.0))
    assert np.allclose(approx, 0.5)


def test_activation_sigmoid_and_invalid():
    vec = np.array([0.0, 1.0])
    out = activation_apply(vec, kind="sigmoid")
    expected = np.array([0.5, 1.0 / (1.0 + math.exp(-1.0))])
    assert np.allclose(out, expected)
    with pytest.raises(ValueError):
        activation_apply(vec, kind="relu")


def test_activation_erf_fallback(monkeypatch):
    import qargus.activations as activations

    monkeypatch.setattr(activations, "_HAS_NP_ERF", False)
    vec = np.array([0.0, 0.5])
    out = activations.activation_apply(vec, kind="erf")
    expected = np.array([math.erf(0.0), math.erf(0.5)])
    assert np.allclose(out, expected)


def test_encoded_tensor_shape_validation():
    encoding = VectorEncoding.from_vector(
        np.array([1.0, 0.0]),
        resources=ResourceEstimate(),
        success=SuccessModel(),
    )
    with pytest.raises(ValueError):
        EncodedTensor(encoding=encoding, shape=(0,))
    with pytest.raises(ValueError):
        EncodedTensor(encoding=encoding, shape=(3,))


def test_pad_1d_validation_and_epsilon():
    class DummyEncoding:
        dimension = 4
        resources = ResourceEstimate()
        success = SuccessModel()
        epsilon = 0.0

        @staticmethod
        def semantic_state():
            return np.ones((2, 2))

    encoded_2d = EncodedTensor(encoding=DummyEncoding(), shape=(2, 2))
    with pytest.raises(ValueError):
        pad_1d(encoded_2d, 4)
    encoded_1d = encoded_from_array(np.ones(3))
    with pytest.raises(ValueError):
        pad_1d(encoded_1d, 2)
    trace = Trace()
    result = pad_1d(encoded_1d, 3, trace=trace, epsilon=0.1)
    assert result.encoded.dimension == 3
    assert math.isclose(result.trace.epsilon, 0.1)


def test_l2_pool_1d_validation():
    encoded = encoded_from_array(np.ones(4))
    with pytest.raises(ValueError):
        l2_pool_1d(encoded, pool=0)
    with pytest.raises(ValueError):
        l2_pool_1d(encoded_from_array(np.ones(3)), pool=2)


def test_vector_sum_validation():
    left = encoded_from_array(np.ones(2))
    right = encoded_from_array(np.ones(3))
    with pytest.raises(ValueError):
        vector_sum(left, right)
    with pytest.raises(ValueError):
        vector_sum(left, encoded_from_array(np.ones(2)), tau=-0.1)


def test_apply_block_encoding_output_dim_validation():
    mat = np.eye(2)
    block = BlockEncoding(
        op=DenseOperator(mat),
        alpha=1.0,
        resources=ResourceEstimate(),
        success=SuccessModel(),
    )
    encoded = encoded_from_array(np.array([1.0, 0.0]))
    with pytest.raises(ValueError):
        apply_block_encoding(encoded, block, out_shape=(2,), output_dim=0)
    with pytest.raises(ValueError):
        apply_block_encoding(encoded, block, out_shape=(2,), output_dim=3)
    result = apply_block_encoding(encoded, block, out_shape=(1,), output_dim=1)
    assert result.encoded.dimension == 1


def test_estimate_count_validation():
    with pytest.raises(ValueError):
        estimate_count([], precision_bits=1)
    with pytest.raises(ValueError):
        estimate_count([True], precision_bits=0)


def test_resource_estimation_validation():
    with pytest.raises(ValueError):
        _qubits_for_dim(0)
    with pytest.raises(ValueError):
        estimate_block_encoding_resources(0, 1)
    res = estimate_block_encoding_resources(2, 2, nonzeros=0)
    assert res.t_count == 1
    with pytest.raises(ValueError):
        estimate_dense_resources(np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        estimate_activation_resources(4, degree=0)


def test_residual_block_validation_and_resources():
    conv_block = BlockEncoding(
        op=DenseOperator(np.eye(4)),
        alpha=1.0,
        resources=ResourceEstimate(),
        success=SuccessModel(),
    )
    with pytest.raises(ValueError):
        ResidualBlock(conv_block=conv_block, input_shape=(1, 2))
    with pytest.raises(ValueError):
        ResidualBlock(conv_block=conv_block, input_shape=(1, 2, 2), skip_tau=1.5)
    with pytest.raises(ValueError):
        ResidualBlock(conv_block=conv_block, input_shape=(1, 2, 2), activation_error=-0.1)
    with pytest.raises(ValueError):
        ResidualBlock(conv_block=conv_block, input_shape=(1, 2, 2), normalize_error=-0.1)
    block = ResidualBlock(
        conv_block=conv_block,
        input_shape=(1, 2, 2),
        approx_coeffs=np.array([0.1, 0.2, 0.3]),
    )
    resources = block._activation_resources(4)
    assert resources.t_count > 0


def test_model_config_validation():
    with pytest.raises(ValueError):
        QuantumResNet(QuantumResNetConfig(input_shape=(1, 2, 2), channels=2))
    with pytest.raises(ValueError):
        QuantumResNet(QuantumResNetConfig(input_shape=(1, 2, 2), output_mode="bad"))
    with pytest.raises(ValueError):
        QuantumResNet(QuantumResNetConfig(input_shape=(1, 2, 2), activation_error=-0.1))
    with pytest.raises(ValueError):
        QuantumResNet(QuantumResNetConfig(input_shape=(1, 2, 2), normalize_error=-0.1))
    with pytest.raises(ValueError):
        QuantumResNet(QuantumResNetConfig(input_shape=(1, 2, 2), pool_error=-0.1))
    with pytest.raises(ValueError):
        QuantumResNet(QuantumResNetConfig(input_shape=(1, 2, 2), square_error=-0.1))
    with pytest.raises(ValueError):
        QuantumResNet(
            QuantumResNetConfig(
                input_shape=(1, 2, 2),
                num_classes=0,
                output_mode="regime3",
                num_blocks=0,
                channels=1,
                kernel_size=1,
            )
        )
    with pytest.raises(ValueError):
        QuantumResNet(
            QuantumResNetConfig(
                input_shape=(1, 2, 2),
                num_classes=2,
                output_mode="regime3",
                pool_size=0,
                num_blocks=0,
                channels=1,
                kernel_size=1,
            )
        )
    with pytest.raises(ValueError):
        QuantumResNet(
            QuantumResNetConfig(
                input_shape=(1, 2, 2),
                num_classes=3,
                output_mode="regime3",
                pool_size=2,
                num_blocks=0,
                channels=1,
                kernel_size=1,
            )
        )


def test_model_forward_regime3_no_padding():
    config = QuantumResNetConfig(
        input_shape=(1, 2, 2),
        num_classes=2,
        num_blocks=0,
        channels=1,
        kernel_size=1,
        output_mode="regime3",
    )
    model = QuantumResNet(config)
    x = np.array([[[1.0, 0.0], [0.0, 0.0]]])
    result = model.forward(x)
    assert result.probabilities.shape == (2,)
    assert math.isclose(float(np.sum(result.probabilities)), 1.0)


def test_semantic_validation_paths():
    with pytest.raises(ValueError):
        _l2_normalize(np.zeros(2))

    class DummyModel:
        blocks = [object()]
        output_mode = "regime3"
        output_padded_dim = 2
        output_pool_size = 1

    with pytest.raises(ValueError):
        build_semantic_program(DummyModel())

    class DummyClassifierModel:
        blocks = []
        output_mode = "classifier"
        classifier = None
        classifier_out_dim = None

    with pytest.raises(ValueError):
        build_semantic_program(DummyClassifierModel())

    step = Regime3OutputStep(output_padded_dim=2, output_pool_size=1)
    state = StateVector(np.ones(4))
    report = RunReport()
    with pytest.raises(ValueError):
        step.run_semantic(state, report)

    step = Regime3OutputStep(output_padded_dim=4, output_pool_size=3)
    state = StateVector(np.ones(4))
    report = RunReport()
    with pytest.raises(ValueError):
        step.run_semantic(state, report)


def test_run_semantic_model_input_shape_validation():
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
        run_semantic_model(model, np.ones((2, 2)))
