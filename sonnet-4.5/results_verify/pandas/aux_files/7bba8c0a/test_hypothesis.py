import numpy as np
from hypothesis import given, strategies as st, assume, settings
import pandas.util

@given(st.lists(st.text(), min_size=2, max_size=50))
@settings(max_examples=100)
def test_hash_array_categorize_parameter(lst):
    assume(len(set(lst)) < len(lst))  # Ensure there are duplicates
    arr = np.array(lst, dtype=object)
    hash_with_categorize = pandas.util.hash_array(arr, categorize=True)
    hash_without_categorize = pandas.util.hash_array(arr, categorize=False)

    assert np.array_equal(hash_with_categorize, hash_without_categorize), \
        f"Hashes differ for input {lst!r}"

# Run the test
if __name__ == "__main__":
    try:
        test_hash_array_categorize_parameter()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")

    # Test the specific failing case
    print("\nTesting specific failing case: ['', '', '\\x00']")
    lst = ['', '', '\x00']
    arr = np.array(lst, dtype=object)
    hash_with_categorize = pandas.util.hash_array(arr, categorize=True)
    hash_without_categorize = pandas.util.hash_array(arr, categorize=False)

    print(f"categorize=True:  {hash_with_categorize}")
    print(f"categorize=False: {hash_without_categorize}")
    print(f"Arrays equal? {np.array_equal(hash_with_categorize, hash_without_categorize)}")