import numpy as np
import numpy.ma as ma
from hypothesis import given, strategies as st

@given(st.integers(min_value=2, max_value=10), st.integers(min_value=2, max_value=10))
def test_compress_rows_shape_inconsistency_when_all_masked(rows, cols):
    data = np.zeros((rows, cols))
    mask = np.ones((rows, cols), dtype=bool)

    arr = ma.array(data, mask=mask)
    result = ma.compress_rows(arr)

    assert result.ndim == 2
    assert result.shape == (0, cols)

@given(st.integers(min_value=2, max_value=10), st.integers(min_value=2, max_value=10))
def test_compress_cols_shape_inconsistency_when_all_masked(rows, cols):
    data = np.zeros((rows, cols))
    mask = np.ones((rows, cols), dtype=bool)

    arr = ma.array(data, mask=mask)
    result = ma.compress_cols(arr)

    assert result.ndim == 2
    assert result.shape == (rows, 0)

if __name__ == "__main__":
    # Run the tests
    try:
        test_compress_rows_shape_inconsistency_when_all_masked()
        print("test_compress_rows_shape_inconsistency_when_all_masked passed")
    except AssertionError as e:
        print(f"test_compress_rows_shape_inconsistency_when_all_masked failed: {e}")

    try:
        test_compress_cols_shape_inconsistency_when_all_masked()
        print("test_compress_cols_shape_inconsistency_when_all_masked passed")
    except AssertionError as e:
        print(f"test_compress_cols_shape_inconsistency_when_all_masked failed: {e}")