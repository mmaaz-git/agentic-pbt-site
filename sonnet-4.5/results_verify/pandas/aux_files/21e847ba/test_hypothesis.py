import numpy as np
from hypothesis import given, strategies as st, settings
from pandas.core.util.hashing import hash_array


@settings(max_examples=500)
@given(st.lists(st.text(min_size=0, max_size=10), min_size=1, max_size=20))
def test_hash_array_categorize_equivalence_strings(data):
    arr = np.array(data, dtype=object)
    hash_with_categorize = hash_array(arr, categorize=True)
    hash_without_categorize = hash_array(arr, categorize=False)

    assert np.array_equal(hash_with_categorize, hash_without_categorize)

# Run the test
if __name__ == "__main__":
    test_hash_array_categorize_equivalence_strings()