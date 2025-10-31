import numpy as np
from hypothesis import given, strategies as st, settings


@given(st.text(min_size=16, max_size=16))
@settings(max_examples=500)
def test_hash_array_different_hash_keys(hash_key):
    from pandas.core.util.hashing import hash_array

    arr = np.array([1, 2, 3])
    default_hash = hash_array(arr, hash_key="0123456789123456")
    custom_hash = hash_array(arr, hash_key=hash_key)

    if hash_key == "0123456789123456":
        assert np.array_equal(default_hash, custom_hash)
    else:
        assert not np.array_equal(default_hash, custom_hash)


if __name__ == "__main__":
    test_hash_array_different_hash_keys()