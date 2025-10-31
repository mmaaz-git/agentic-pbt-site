from hypothesis import given, strategies as st, settings, find
import numpy as np
from pandas.core.util.hashing import hash_array

@given(st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=50))
@settings(max_examples=100)
def test_hash_array_consistent_across_dtypes(values):
    arr_int32 = np.array(values, dtype=np.int32)
    arr_int64 = np.array(values, dtype=np.int64)

    hash_int32 = hash_array(arr_int32)
    hash_int64 = hash_array(arr_int64)

    assert np.array_equal(hash_int32, hash_int64), f"Failed for values: {values}"

if __name__ == "__main__":
    # Test with the specific failing case
    print("Testing with the specific failing case: [-1]")
    values = [-1]
    arr_int32 = np.array(values, dtype=np.int32)
    arr_int64 = np.array(values, dtype=np.int64)

    hash_int32 = hash_array(arr_int32)
    hash_int64 = hash_array(arr_int64)

    try:
        assert np.array_equal(hash_int32, hash_int64), f"Failed for values: {values}"
        print("Test passed for [-1]")
    except AssertionError as e:
        print(f"Test failed for [-1]: Hashes not equal")
        print(f"  int32 hash: {hash_int32}")
        print(f"  int64 hash: {hash_int64}")

    # Run the property-based test
    print("\nRunning property-based test with multiple examples...")
    try:
        failing_example = find(st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=50),
                               lambda values: not np.array_equal(hash_array(np.array(values, dtype=np.int32)),
                                                                  hash_array(np.array(values, dtype=np.int64))))
        print(f"Found failing example: {failing_example}")
    except Exception as e:
        print(f"No failures found or error occurred: {e}")