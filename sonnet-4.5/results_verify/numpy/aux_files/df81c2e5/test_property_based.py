import numpy as np
from numpy.matrixlib import matrix
from hypothesis import given, strategies as st
import pytest


@given(st.integers(1, 5), st.integers(2, 5))
def test_3d_array_vs_list_consistency(n, m):
    shape_3d = (1, n, m)

    arr_3d = np.zeros(shape_3d)
    list_3d = [[[0.0] * m for _ in range(n)]]

    arr_result = None
    arr_error = None
    try:
        arr_result = matrix(arr_3d)
    except ValueError as e:
        arr_error = str(e)

    list_result = None
    list_error = None
    try:
        list_result = matrix(list_3d)
    except ValueError as e:
        list_error = str(e)

    if arr_result is None and list_result is None:
        assert arr_error == list_error, f"Both failed but with different errors: '{arr_error}' vs '{list_error}'"
    elif arr_result is not None and list_result is not None:
        assert arr_result.shape == list_result.shape, "Both succeeded but with different shapes"
    else:
        arr_status = 'succeeded' if arr_result is not None else 'failed'
        list_status = 'succeeded' if list_result is not None else 'failed'
        print(f"Found inconsistency at n={n}, m={m} (shape {shape_3d}): array {arr_status}, list {list_status}")
        pytest.fail(f"Inconsistent behavior: array {arr_status}, list {list_status}")

# Run the test
test_3d_array_vs_list_consistency()