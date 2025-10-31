from hypothesis import given, strategies as st, settings

@given(st.integers(min_value=2, max_value=100))
@settings(max_examples=500, deadline=None)
def test_lru_update_preserves_size(maxsize):
    from dask.dataframe.dask_expr._util import LRU

    lru = LRU(maxsize)

    # Fill the cache to capacity
    for i in range(maxsize):
        lru[f'key_{i}'] = i

    assert len(lru) == maxsize

    # Update an existing key (not the first one)
    # We pick a middle key to ensure it's not the oldest
    update_key = f'key_{maxsize // 2}'
    lru[update_key] = 999

    assert len(lru) == maxsize, f"Updating existing key should preserve size {maxsize}, got {len(lru)}"

if __name__ == "__main__":
    # Run the test
    test_lru_update_preserves_size()