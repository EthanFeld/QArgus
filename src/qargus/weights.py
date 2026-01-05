from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

from blockflow import BlockEncoding, ResourceEstimate, SuccessModel

from .operators import Convolution2DOperator, DenseOperator
from .resource_estimation import estimate_conv_resources, estimate_dense_resources


def conv_filters_from_keras(kernel: np.ndarray) -> np.ndarray:
    """
    Convert a Keras Conv2D kernel (H, W, in, out) to QArgus format (out, in, H, W).
    """
    kernel = np.asarray(kernel)
    if kernel.ndim != 4:
        raise ValueError("keras conv kernel must be 4D")
    return np.transpose(kernel, (3, 2, 0, 1))


def dense_weights_from_keras(kernel: np.ndarray) -> np.ndarray:
    """
    Convert a Keras Dense kernel (in, out) to QArgus format (out, in).
    """
    kernel = np.asarray(kernel)
    if kernel.ndim != 2:
        raise ValueError("keras dense kernel must be 2D")
    return kernel.T


def sklearn_linear_weights(estimator: object) -> np.ndarray:
    """
    Extract linear weights from a sklearn estimator with coef_.
    Bias terms (intercept_) are ignored.
    """
    if not hasattr(estimator, "coef_"):
        raise ValueError("estimator must expose coef_")
    weights = np.asarray(getattr(estimator, "coef_"))
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
    if weights.ndim != 2:
        raise ValueError("coef_ must be 1D or 2D")
    return weights


def _build_conv_block(
    filters: np.ndarray,
    *,
    input_shape: tuple[int, int, int],
    stride: int,
    padding: int,
) -> BlockEncoding:
    conv_op = Convolution2DOperator(
        filters=filters,
        input_shape=input_shape,
        stride=stride,
        padding=padding,
    )
    return BlockEncoding(
        op=conv_op,
        alpha=float(np.linalg.norm(filters)),
        resources=estimate_conv_resources(conv_op),
        success=SuccessModel(),
    )


def load_conv_filters(model: object, filters_list: Sequence[np.ndarray]) -> None:
    """
    Replace convolution filters for each residual block.
    """
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise ValueError("model must expose blocks")
    if len(filters_list) != len(blocks):
        raise ValueError("filters_list length must match number of blocks")
    for block, filters in zip(blocks, filters_list):
        conv_block = getattr(block, "conv_block", None)
        if conv_block is None or not hasattr(conv_block, "op"):
            raise ValueError("block is missing a convolution operator")
        op = conv_block.op
        if not isinstance(op, Convolution2DOperator):
            raise ValueError("block convolution operator must be Convolution2DOperator")
        filters = np.asarray(filters)
        if filters.ndim != 4:
            raise ValueError("filters must be a 4D array")
        block.conv_block = _build_conv_block(
            filters,
            input_shape=op.input_shape,
            stride=op.stride,
            padding=op.padding,
        )


def load_classifier_weights(model: object, weights: np.ndarray) -> None:
    """
    Load a dense classifier weight matrix into a QuantumResNet classifier head.
    """
    if getattr(model, "output_mode", None) != "classifier":
        raise ValueError("model.output_mode must be 'classifier' to load dense weights")
    weights = np.asarray(weights)
    if weights.ndim != 2:
        raise ValueError("weights must be a 2D array")
    num_classes = int(getattr(model.config, "num_classes"))
    flatten_dim = int(getattr(model, "flatten_dim"))
    if weights.shape == (flatten_dim, num_classes):
        weights = weights.T
    if weights.shape != (num_classes, flatten_dim):
        raise ValueError("weights shape must be (num_classes, flatten_dim)")
    square_dim = max(flatten_dim, num_classes)
    square_mat = np.zeros((square_dim, square_dim), dtype=weights.dtype)
    square_mat[:num_classes, :flatten_dim] = weights
    dense_block = BlockEncoding(
        op=DenseOperator(square_mat),
        alpha=float(np.linalg.norm(square_mat)),
        resources=estimate_dense_resources(square_mat),
        success=SuccessModel(),
    )
    model.classifier = dense_block
    model.classifier_out_dim = num_classes


def load_sklearn_classifier(model: object, estimator: object) -> None:
    """
    Load weights from a sklearn linear estimator into a classifier head.
    """
    weights = sklearn_linear_weights(estimator)
    load_classifier_weights(model, weights)


def _layer_kernel(layer: object) -> np.ndarray:
    if hasattr(layer, "get_weights"):
        weights = layer.get_weights()
        if not weights:
            raise ValueError("layer.get_weights returned no weights")
        return np.asarray(weights[0])
    if hasattr(layer, "kernel"):
        return np.asarray(getattr(layer, "kernel"))
    raise ValueError("layer does not expose a kernel")


def _is_layer_type(layer: object, name: str) -> bool:
    class_name = layer.__class__.__name__.lower()
    return name in class_name


def load_keras_model(
    model: object,
    keras_model: object,
    *,
    conv_layer_names: Optional[Iterable[str]] = None,
    dense_layer_name: Optional[str] = None,
) -> None:
    """
    Load Conv2D/Dense weights from a Keras model. Bias terms are ignored.
    """
    layers = getattr(keras_model, "layers", None)
    if layers is None:
        raise ValueError("keras_model must expose layers")
    if conv_layer_names is None:
        conv_layers = [layer for layer in layers if _is_layer_type(layer, "conv2d")]
    else:
        conv_names = set(conv_layer_names)
        conv_layers = [layer for layer in layers if getattr(layer, "name", None) in conv_names]
    if len(conv_layers) < len(getattr(model, "blocks", [])):
        raise ValueError("not enough Conv2D layers to populate model blocks")
    conv_layers = conv_layers[: len(getattr(model, "blocks", []))]
    conv_filters = [conv_filters_from_keras(_layer_kernel(layer)) for layer in conv_layers]
    load_conv_filters(model, conv_filters)

    if getattr(model, "output_mode", None) != "classifier":
        return
    if dense_layer_name is None:
        dense_layers = [layer for layer in layers if _is_layer_type(layer, "dense")]
        if not dense_layers:
            return
        dense_layer = dense_layers[-1]
    else:
        dense_layer = None
        for layer in layers:
            if getattr(layer, "name", None) == dense_layer_name:
                dense_layer = layer
                break
        if dense_layer is None:
            raise ValueError("dense_layer_name not found")
    dense_weights = dense_weights_from_keras(_layer_kernel(dense_layer))
    load_classifier_weights(model, dense_weights)
