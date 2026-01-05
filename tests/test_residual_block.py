import numpy as np

from blockflow import BlockEncoding, ResourceEstimate, SuccessModel

from qargus.blocks import ResidualBlock
from qargus.encoding import apply_block_encoding, encoded_from_array, l2_normalize, vector_sum
from qargus.operators import DenseOperator


def test_residual_block_applies_skip_and_normalize():
    input_shape = (1, 1, 2)
    x = np.array([[[1.0, 2.0]]])
    encoded = encoded_from_array(x)
    normalized = l2_normalize(encoded)

    mat = np.array([[0.0, 1.0], [1.0, 0.0]])
    conv_block = BlockEncoding(
        op=DenseOperator(mat),
        alpha=float(np.linalg.norm(mat)),
        resources=ResourceEstimate(),
        success=SuccessModel(),
    )
    block = ResidualBlock(
        conv_block=conv_block,
        input_shape=input_shape,
        activation="identity",
        skip_tau=0.5,
    )
    output = block.apply(normalized.encoded, trace=normalized.trace)

    conv = apply_block_encoding(normalized.encoded, conv_block, out_shape=input_shape, trace=normalized.trace)
    summed = vector_sum(normalized.encoded, conv.encoded, tau=0.5, trace=conv.trace)
    expected = l2_normalize(summed.encoded, trace=summed.trace)

    assert np.allclose(
        output.encoded.encoding.semantic_state(),
        expected.encoded.encoding.semantic_state(),
    )
