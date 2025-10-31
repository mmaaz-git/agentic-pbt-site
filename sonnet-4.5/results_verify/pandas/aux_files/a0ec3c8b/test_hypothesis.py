from hypothesis import given, strategies as st, example, seed
import numpy as np
from pandas.core.util.hashing import hash_array

@given(st.lists(st.text(min_size=0, max_size=5), min_size=10, max_size=50))
@example(['', '', '', '', '', '', '', '', '', '\x00'])  # The failing case from the bug report
def test_hash_array_strings_with_duplicates(values):
    arr = np.array(values, dtype=object)
    hash_categorized = hash_array(arr, categorize=True)
    hash_uncategorized = hash_array(arr, categorize=False)

    # Check if they're equal
    if not np.array_equal(hash_categorized, hash_uncategorized):
        print(f"\nFailing input: {values}")
        print(f"With categorize=True:  {hash_categorized}")
        print(f"With categorize=False: {hash_uncategorized}")

        # Check for differences
        for i in range(len(values)):
            if hash_categorized[i] != hash_uncategorized[i]:
                print(f"  Index {i}: value={repr(values[i])}, cat={hash_categorized[i]}, uncat={hash_uncategorized[i]}")

        assert False, "Hash values differ between categorize=True and categorize=False"

# Run the test with a fixed seed for reproducibility
print("Running hypothesis test...")
try:
    test_hash_array_strings_with_duplicates()
    print("Test passed for all generated cases!")
except AssertionError as e:
    print(f"Test failed: {e}")