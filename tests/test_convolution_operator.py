import numpy as np

from qargus.operators import Convolution2DOperator


def test_convolution2d_operator_apply():
    x = np.array(
        [
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        ]
    )
    filt = np.array([[[[1.0, 0.0], [0.0, -1.0]]]])
    op = Convolution2DOperator(filters=filt, input_shape=(1, 3, 3), stride=1, padding=0)
    out = op.apply(x.reshape(-1))
    expected = np.array([-4.0, -4.0, -4.0, -4.0])
    assert np.allclose(out, expected)
