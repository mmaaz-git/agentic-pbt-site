from hypothesis import given, strategies as st, settings
import numpy as np
from pandas.core.util.hashing import hash_array

@given(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1))
@settings(max_examples=500)
def test_hash_equal_values_have_equal_hashes(values):
    arr = np.array(values)

    for i in range(len(arr)):
        if arr[i] == 0.0:
            arr_pos = arr.copy()
            arr_pos[i] = 0.0
            arr_neg = arr.copy()
            arr_neg[i] = -0.0

            hash_pos = hash_array(arr_pos)
            hash_neg = hash_array(arr_neg)

            assert np.array_equal(hash_pos, hash_neg), \
                f"Equal arrays should have equal hashes: {arr_pos} vs {arr_neg}"

if __name__ == "__main__":
    test_hash_equal_values_have_equal_hashes()
    print("Test passed!")