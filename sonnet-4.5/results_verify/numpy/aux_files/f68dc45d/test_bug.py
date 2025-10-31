import numpy as np
import numpy.ma as ma
from hypothesis import given, strategies as st, settings
from hypothesis.extra import numpy as npst

# Test case from the bug report
@st.composite
def masked_2d_arrays(draw):
    rows = draw(st.integers(min_value=1, max_value=5))
    cols = draw(st.integers(min_value=1, max_value=5))
    data = draw(npst.arrays(dtype=np.float64, shape=(rows, cols),
                           elements=st.floats(allow_nan=False, allow_infinity=False,
                                            min_value=-100, max_value=100)))
    mask = draw(npst.arrays(dtype=bool, shape=(rows, cols)))
    return ma.array(data, mask=mask)

@given(masked_2d_arrays())
@settings(max_examples=500)
def test_compress_rows_preserves_2d_structure(arr):
    result = ma.compress_rows(arr)
    assert result.ndim == 2, f"Expected 2D array, got {result.ndim}D with shape {result.shape}"

# Run the test
print("Running hypothesis test...")
try:
    test_compress_rows_preserves_2d_structure()
    print("Hypothesis test PASSED")
except Exception as e:
    print(f"Hypothesis test FAILED: {e}")

# Manual reproduction
print("\n=== Manual reproduction ===")
arr = ma.array([[999]], mask=[[True]])

result_rows = ma.compress_rows(arr)
result_cols = ma.compress_cols(arr)
result_rowcols = ma.compress_rowcols(arr)

print(f"Input shape: {arr.shape} (2D), ndim={arr.ndim}")
print(f"compress_rows result: shape={result_rows.shape}, ndim={result_rows.ndim}, type={type(result_rows)}")
print(f"compress_cols result: shape={result_cols.shape}, ndim={result_cols.ndim}, type={type(result_cols)}")
print(f"compress_rowcols result: shape={result_rowcols.shape}, ndim={result_rowcols.ndim}, type={type(result_rowcols)}")

print(f"\nResult_rows content: {result_rows}")
print(f"Result_cols content: {result_cols}")
print(f"Result_rowcols content: {result_rowcols}")

# Multi-column example
print("\n=== Multi-column example ===")
arr2 = ma.array([[1, 2], [3, 4]], mask=[[True, True], [True, True]])
result2_rows = ma.compress_rows(arr2)
result2_cols = ma.compress_cols(arr2)
result2_rowcols = ma.compress_rowcols(arr2)

print(f"Multi-column input: shape={arr2.shape}, ndim={arr2.ndim}")
print(f"compress_rows result: shape={result2_rows.shape}, ndim={result2_rows.ndim}")
print(f"compress_cols result: shape={result2_cols.shape}, ndim={result2_cols.ndim}")
print(f"compress_rowcols result: shape={result2_rowcols.shape}, ndim={result2_rowcols.ndim}")

# Test with partially masked arrays
print("\n=== Partially masked array ===")
arr3 = ma.array([[1, 2], [3, 4]], mask=[[True, False], [False, False]])
result3_rows = ma.compress_rows(arr3)
result3_cols = ma.compress_cols(arr3)
result3_rowcols = ma.compress_rowcols(arr3)

print(f"Partially masked input: shape={arr3.shape}, ndim={arr3.ndim}")
print(f"compress_rows result: shape={result3_rows.shape}, ndim={result3_rows.ndim}, content={result3_rows}")
print(f"compress_cols result: shape={result3_cols.shape}, ndim={result3_cols.ndim}, content={result3_cols}")
print(f"compress_rowcols result: shape={result3_rowcols.shape}, ndim={result3_rowcols.ndim}, content={result3_rowcols}")

# Check what happens with downstream code
print("\n=== Downstream code test ===")
arr4 = ma.array([[999]], mask=[[True]])
try:
    result = ma.compress_rows(arr4)
    num_cols = result.shape[1]
    print(f"Successfully accessed shape[1]: {num_cols}")
except IndexError as e:
    print(f"IndexError when accessing shape[1]: {e}")

# Check regular numpy arrays behavior for comparison
print("\n=== Regular numpy array indexing comparison ===")
regular_2d = np.array([[1, 2], [3, 4]])
empty_2d = regular_2d[np.array([], dtype=int)]
print(f"Regular 2D array shape: {regular_2d.shape}, ndim: {regular_2d.ndim}")
print(f"Empty slice of 2D array shape: {empty_2d.shape}, ndim: {empty_2d.ndim}")