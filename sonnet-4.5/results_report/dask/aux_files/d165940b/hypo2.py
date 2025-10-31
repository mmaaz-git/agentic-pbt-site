from hypothesis import given, strategies as st, settings, assume
from dask.dataframe.dask_expr._util import LRU

@given(st.integers(min_value=2, max_value=10))
@settings(max_examples=200)
def test_lru_update_evicts_unnecessarily(maxsize):
    """Test that updating an existing key in a full cache unnecessarily evicts another key."""
    lru = LRU(maxsize)

    # Fill the cache completely
    for i in range(maxsize):
        lru[i] = i * 2

    # Remember initial size
    initial_size = len(lru)
    assert initial_size == maxsize

    # Update an existing key (not adding a new one)
    lru[0] = 999  # Update key 0 with new value

    # The cache should still have the same size since we only updated
    final_size = len(lru)

    # Bug: The cache size decreases because it evicts before checking if key exists
    assert final_size == initial_size, f"Cache size changed from {initial_size} to {final_size} after updating existing key (maxsize={maxsize})"

if __name__ == "__main__":
    test_lru_update_evicts_unnecessarily()