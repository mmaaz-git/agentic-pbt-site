import numpy as np
import numpy.ma as ma
from hypothesis import given, strategies as st, settings

# Test from bug report
@settings(max_examples=300)
@given(
    st.integers(min_value=2, max_value=10),
    st.integers(min_value=2, max_value=10)
)
def test_compress_rowcols_ndim_consistency(rows, cols):
    arr = np.arange(rows * cols).reshape(rows, cols).astype(float)
    all_masked = ma.array(arr, mask=np.ones((rows, cols), dtype=bool))

    result = ma.compress_rowcols(all_masked, 0)

    assert result.ndim == 2, f"Expected 2D array, got {result.ndim}D with shape {result.shape}"

# Run the reproduction
print("=" * 50)
print("Testing the bug reproduction...")
print("=" * 50)

arr = np.ones((3, 4))
all_masked = ma.array(arr, mask=np.ones((3, 4), dtype=bool))

result = ma.compress_rowcols(all_masked, axis=0)

print(f"Input shape: {all_masked.shape}")
print(f"Result shape: {result.shape}")
print(f"Result ndim: {result.ndim}")

try:
    assert result.ndim == 2, f"Expected 2D, got {result.ndim}D"
    print("Assertion passed: Result is 2D")
except AssertionError as e:
    print(f"AssertionError: {e}")

print("\n" + "=" * 50)
print("Testing with partially masked array...")
print("=" * 50)

# Test with partially masked array
arr2 = np.ones((3, 4))
partial_mask = np.array([[True, True, True, True],
                         [False, False, False, False],
                         [True, True, True, True]])
partial_masked = ma.array(arr2, mask=partial_mask)

result2 = ma.compress_rowcols(partial_masked, axis=0)
print(f"Partially masked input shape: {partial_masked.shape}")
print(f"Partially masked result shape: {result2.shape}")
print(f"Partially masked result ndim: {result2.ndim}")

print("\n" + "=" * 50)
print("Testing axis=1 (columns)...")
print("=" * 50)

# Test axis=1 with all masked
result3 = ma.compress_rowcols(all_masked, axis=1)
print(f"All masked, axis=1 result shape: {result3.shape}")
print(f"All masked, axis=1 result ndim: {result3.ndim}")

# Test axis=1 with partial mask
partial_mask_cols = np.array([[True, False, True, False],
                              [True, False, True, False],
                              [True, False, True, False]])
partial_masked_cols = ma.array(arr2, mask=partial_mask_cols)
result4 = ma.compress_rowcols(partial_masked_cols, axis=1)
print(f"Partially masked, axis=1 input shape: {partial_masked_cols.shape}")
print(f"Partially masked, axis=1 result shape: {result4.shape}")
print(f"Partially masked, axis=1 result ndim: {result4.ndim}")

print("\n" + "=" * 50)
print("Testing axis=None (both rows and columns)...")
print("=" * 50)

result5 = ma.compress_rowcols(all_masked, axis=None)
print(f"All masked, axis=None result shape: {result5.shape}")
print(f"All masked, axis=None result ndim: {result5.ndim}")

print("\n" + "=" * 50)
print("Running hypothesis test...")
print("=" * 50)

try:
    test_compress_rowcols_ndim_consistency(2, 2)
    print("Hypothesis test passed for (2, 2)")
except AssertionError as e:
    print(f"Hypothesis test failed for (2, 2): {e}")