import numpy as np
from pandas.core.util.hashing import hash_array

data = ['', '\x00']
arr = np.array(data, dtype=object)

hash_with_categorize = hash_array(arr, categorize=True)
hash_without_categorize = hash_array(arr, categorize=False)

print("Input:", data)
print("Hash with categorize=True: ", hash_with_categorize)
print("Hash with categorize=False:", hash_without_categorize)

try:
    assert hash_with_categorize[0] == hash_with_categorize[1], \
        "BUG: Empty string and null byte have SAME hash with categorize=True"
    print("\n✓ Confirmed: Empty string and null byte have SAME hash with categorize=True")
except AssertionError as e:
    print(f"\n✗ Assertion failed: {e}")

try:
    assert hash_without_categorize[0] != hash_without_categorize[1], \
        "With categorize=False, they correctly have DIFFERENT hashes"
    print("✓ Confirmed: With categorize=False, they correctly have DIFFERENT hashes")
except AssertionError as e:
    print(f"✗ Assertion failed: {e}")