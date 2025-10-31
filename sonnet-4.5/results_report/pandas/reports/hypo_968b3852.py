import numpy as np
from hypothesis import given, strategies as st
from pandas.core.util.hashing import hash_array


@given(st.lists(st.integers(), min_size=1, max_size=50))
def test_hash_array_key_ignored_for_numeric_arrays(values):
    arr = np.array(values)
    hash_key1 = '0' * 16
    hash_key2 = '1' * 16

    hash1 = hash_array(arr, hash_key=hash_key1, categorize=False)
    hash2 = hash_array(arr, hash_key=hash_key2, categorize=False)

    assert not np.array_equal(hash1, hash2), \
        f"Different hash keys should produce different hashes for numeric arrays. Input: {values}"


if __name__ == "__main__":
    test_hash_array_key_ignored_for_numeric_arrays()