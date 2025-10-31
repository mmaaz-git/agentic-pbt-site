from hypothesis import given, strategies as st
from pandas.core.arrays.sparse import SparseArray
import numpy as np

@given(st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=100))
def test_cumsum_matches_dense(values):
    """Test that SparseArray.cumsum() matches the dense array cumsum."""
    sparse_arr = SparseArray(values, fill_value=0)

    # Calculate cumsum on sparse array
    try:
        sparse_cumsum = sparse_arr.cumsum().to_dense()
    except RecursionError:
        print(f"\nRecursionError on input: {values[:5]}{'...' if len(values) > 5 else ''}")
        raise

    # Calculate cumsum on dense array
    dense_cumsum = sparse_arr.to_dense().cumsum()

    # They should be equal
    assert np.array_equal(sparse_cumsum, dense_cumsum), \
        f"cumsum() on sparse should match cumsum() on dense\nSparse: {sparse_cumsum}\nDense: {dense_cumsum}"

if __name__ == "__main__":
    # Run the test with a simple example that fails
    print("Running hypothesis test with minimal failing example...")
    print("Testing with values: [1, 2, 3]")
    try:
        test_cumsum_matches_dense()
    except Exception as e:
        print(f"\nHypothesis test failed with: {e}")
        print("\nTrying direct test with [1, 2, 3]...")
        # Direct test without hypothesis decorator
        values = [1, 2, 3]
        sparse_arr = SparseArray(values, fill_value=0)
        try:
            sparse_cumsum = sparse_arr.cumsum().to_dense()
            print(f"Unexpected success: {sparse_cumsum}")
        except RecursionError:
            print(f"RecursionError on input: {values}")
            import traceback
            print("\nTraceback (last 5 frames):")
            tb = traceback.format_exc()
            lines = tb.split('\n')
            # Show just a portion to avoid overwhelming output
            for line in lines[-15:]:
                if line:
                    print(line)