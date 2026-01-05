from qargus.counting import estimate_count


def test_quantum_counting_estimate():
    marks = [True] * 4 + [False] * 12
    result = estimate_count(marks, precision_bits=6)
    assert result.num_items == 16
    assert result.true_count == 4
    assert abs(result.estimated_count - 4) <= 1


def test_quantum_counting_extremes_and_half():
    precision_bits = 5
    marks_none = [False] * 8
    result_none = estimate_count(marks_none, precision_bits=precision_bits)
    assert result_none.true_count == 0
    assert result_none.estimated_count == 0
    assert result_none.oracle_queries == (2 ** precision_bits - 1)

    marks_all = [True] * 8
    result_all = estimate_count(marks_all, precision_bits=precision_bits)
    assert result_all.true_count == 8
    assert result_all.estimated_count == 8

    marks_half = [True, False] * 4
    result_half = estimate_count(marks_half, precision_bits=precision_bits)
    assert result_half.true_count == 4
    assert result_half.estimated_count == 4
