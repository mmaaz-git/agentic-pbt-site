import numpy as np
from pandas.core.util.hashing import hash_array
from pandas import factorize

# Test various strings with null characters
test_cases = [
    ['', '\x00'],
    ['a', 'a\x00'],
    ['hello', 'hello\x00'],
    ['', '\x00', 'a', 'a\x00'],
    ['\x00', '\x00\x00'],
    ['', ' '],  # empty string vs space
]

for i, arr_vals in enumerate(test_cases):
    print(f"\n=== Test case {i+1}: {[repr(v) for v in arr_vals]} ===")
    arr = np.array(arr_vals, dtype=object)

    hash_cat = hash_array(arr, categorize=True)
    hash_nocat = hash_array(arr, categorize=False)

    codes, categories = factorize(arr, sort=False)

    print(f"Factorize codes: {codes}")
    print(f"Factorize categories: {[repr(c) for c in categories]}")
    print(f"With categorize=True:  {hash_cat}")
    print(f"With categorize=False: {hash_nocat}")

    # Check if hashes are equal
    all_equal = np.array_equal(hash_cat, hash_nocat)
    print(f"Are hashes equal? {all_equal}")

    if not all_equal:
        for j in range(len(arr_vals)):
            if hash_cat[j] != hash_nocat[j]:
                print(f"  DIFF at index {j}: val={repr(arr_vals[j])}, cat={hash_cat[j]}, nocat={hash_nocat[j]}")