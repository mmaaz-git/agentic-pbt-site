import numpy as np
from hypothesis import given, strategies as st, settings
from pandas.core.util.hashing import hash_array


@settings(max_examples=100)
@given(st.lists(st.integers(), min_size=1, max_size=100))
def test_hash_array_different_encodings(values):
    arr = np.array([str(v) for v in values], dtype=object)
    result_utf8 = hash_array(arr, encoding='utf8')
    result_utf16 = hash_array(arr, encoding='utf16')
    assert len(result_utf8) == len(arr)
    assert len(result_utf16) == len(arr)

# Run the test
if __name__ == "__main__":
    test_hash_array_different_encodings()