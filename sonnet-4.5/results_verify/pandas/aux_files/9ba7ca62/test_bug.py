from pandas.core.util.hashing import hash_tuples, hash_array, combine_hash_arrays
import numpy as np
import pytest

print("Testing hash_tuples with empty list:")
print("=" * 40)

# Test the hypothesis test first
def test_hash_tuples_empty():
    with pytest.raises(TypeError, match="Cannot infer number of levels from empty list"):
        hash_tuples([])

try:
    test_hash_tuples_empty()
    print("Hypothesis test PASSED - TypeError was raised as expected")
except Exception as e:
    print(f"Hypothesis test FAILED: {e}")

print("\n" + "=" * 40)
print("Reproducing the bug with the provided code:")
print("=" * 40)

print("\nhash_array with empty array:")
result = hash_array(np.array([], dtype=np.int64))
print(f"  Success: {result}")
print(f"  Type: {type(result)}, dtype: {result.dtype if hasattr(result, 'dtype') else 'N/A'}")

print("\ncombine_hash_arrays with empty iterator:")
result = combine_hash_arrays(iter([]), 0)
print(f"  Success: {result}")
print(f"  Type: {type(result)}, dtype: {result.dtype if hasattr(result, 'dtype') else 'N/A'}")

print("\nhash_tuples with empty list:")
try:
    result = hash_tuples([])
    print(f"  Success: {result}")
    print(f"  Type: {type(result)}, dtype: {result.dtype if hasattr(result, 'dtype') else 'N/A'}")
except TypeError as e:
    print(f"  Failed with TypeError: {e}")
except Exception as e:
    print(f"  Failed with unexpected error: {type(e).__name__}: {e}")