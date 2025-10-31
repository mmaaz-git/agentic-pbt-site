import numpy as np
import numpy.ma as ma
from hypothesis import given, strategies as st

@given(st.integers(min_value=2, max_value=10), st.integers(min_value=2, max_value=10))
def test_compress_rows_shape_inconsistency_when_all_masked(rows, cols):
    data = np.zeros((rows, cols))
    mask = np.ones((rows, cols), dtype=bool)

    arr = ma.array(data, mask=mask)
    result = ma.compress_rows(arr)

    assert result.ndim == 2, f"Expected 2-D array but got {result.ndim}-D array"
    assert result.shape == (0, cols), f"Expected shape (0, {cols}) but got {result.shape}"

@given(st.integers(min_value=2, max_value=10), st.integers(min_value=2, max_value=10))
def test_compress_cols_shape_inconsistency_when_all_masked(rows, cols):
    data = np.zeros((rows, cols))
    mask = np.ones((rows, cols), dtype=bool)

    arr = ma.array(data, mask=mask)
    result = ma.compress_cols(arr)

    assert result.ndim == 2, f"Expected 2-D array but got {result.ndim}-D array"
    assert result.shape == (rows, 0), f"Expected shape ({rows}, 0) but got {result.shape}"

if __name__ == "__main__":
    print("Testing compress_rows shape consistency...")
    test_compress_rows_shape_inconsistency_when_all_masked()

    print("Testing compress_cols shape consistency...")
    test_compress_cols_shape_inconsistency_when_all_masked()