from hypothesis import given, strategies as st, settings
from dask.dataframe.dask_expr._util import LRU

@given(st.integers(min_value=2, max_value=10))
@settings(max_examples=200)
def test_lru_update_evicts_when_it_shouldnt(maxsize):
    """Test that updating an existing key evicts the least recently used item unnecessarily."""
    lru = LRU(maxsize)

    # Fill the cache
    for i in range(maxsize):
        lru[i] = i * 10

    # Track which key is least recently used (should be 0)
    least_recent_key = 0

    # Update the LAST key (which is NOT the least recently used)
    last_key = maxsize - 1
    lru[last_key] = 999

    # BUG: The least recently used key (0) gets evicted even though we're just updating
    # an existing key, not adding a new one
    assert least_recent_key in lru, f"Key {least_recent_key} was evicted when updating existing key {last_key} (maxsize={maxsize})"

if __name__ == "__main__":
    test_lru_update_evicts_when_it_shouldnt()