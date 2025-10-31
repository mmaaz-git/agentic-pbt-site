import pandas.core.arrays as arrays
from hypothesis import given, settings, strategies as st, assume
import sys

@given(st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=30))
@settings(max_examples=200)
def test_sparse_cumsum_monotonic_nonnegative(data):
    assume(all(x >= 0 for x in data))
    sparse = arrays.SparseArray(data)
    result = sparse.cumsum()
    dense_result = result.to_dense()

    for i in range(len(dense_result) - 1):
        assert dense_result[i] <= dense_result[i + 1]

if __name__ == "__main__":
    # Run the test
    try:
        test_sparse_cumsum_monotonic_nonnegative()
        print("Test passed")
    except RecursionError as e:
        print(f"RecursionError occurred: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Other error occurred: {e}")
        sys.exit(1)