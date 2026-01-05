import numpy as np

from blockflow import BlockEncoding, ResourceEstimate, SuccessModel

from qargus.encoding import (
    apply_block_encoding,
    elementwise_square,
    encoded_from_array,
    vector_sum,
)
from qargus.operators import DenseOperator


def test_vector_sum_normalizes():
    left = encoded_from_array(np.array([1.0, 0.0]))
    right = encoded_from_array(np.array([0.0, 1.0]))
    result = vector_sum(left, right, tau=0.5)
    state = result.encoded.encoding.semantic_state()
    expected = np.array([1.0, 1.0]) / np.sqrt(2.0)
    assert np.allclose(state, expected)


def test_elementwise_square_values():
    encoded = encoded_from_array(np.array([1.0, 1.0j]))
    result = elementwise_square(encoded)
    assert np.allclose(result.encoded.encoding.vec, np.array([0.5, 0.5]))


def test_apply_block_encoding_dense():
    mat = np.array([[1.0, 2.0], [3.0, 4.0]])
    block = BlockEncoding(
        op=DenseOperator(mat),
        alpha=float(np.linalg.norm(mat)),
        resources=ResourceEstimate(),
        success=SuccessModel(),
    )
    encoded = encoded_from_array(np.array([1.0, 0.0]))
    result = apply_block_encoding(encoded, block, out_shape=(2,))
    assert np.allclose(result.encoded.encoding.vec, np.array([1.0, 3.0]))
