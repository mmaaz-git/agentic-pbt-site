import numpy as np
import pandas.util

# Test more edge cases
test_cases = [
    (['a', 'a', 'b'], "Regular strings with duplicates"),
    (['', ' '], "Empty string vs space"),
    (['\x00', '\x01'], "Null byte vs 0x01 byte"),
    (['', '\n'], "Empty string vs newline"),
    (['', '\t'], "Empty string vs tab"),
    ([' ', '  '], "Single space vs double space"),
    (['0', 0], "String '0' vs integer 0"),
    (['', None], "Empty string vs None"),
]

for test_input, description in test_cases:
    print(f"\nTest: {description}")
    print(f"Input: {test_input!r}")
    try:
        arr = np.array(test_input, dtype=object)
        hash_cat = pandas.util.hash_array(arr, categorize=True)
        hash_no_cat = pandas.util.hash_array(arr, categorize=False)

        print(f"categorize=True:  {hash_cat}")
        print(f"categorize=False: {hash_no_cat}")
        print(f"Results match: {np.array_equal(hash_cat, hash_no_cat)}")

        # Check uniqueness
        unique_cat = len(set(hash_cat))
        unique_no_cat = len(set(hash_no_cat))
        print(f"Unique hashes - categorize=True: {unique_cat}, categorize=False: {unique_no_cat}")
    except Exception as e:
        print(f"Error: {e}")