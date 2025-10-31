import numpy as np
from pandas.core.util.hashing import hash_array

# Test the specific case mentioned in the bug report
arr = np.array(['', '\x00'], dtype=object)

hash_with_categorize = hash_array(arr, categorize=True)
hash_without_categorize = hash_array(arr, categorize=False)

print("Input:", [repr(v) for v in arr])
print("With categorize=True: ", hash_with_categorize)
print("With categorize=False:", hash_without_categorize)

print("\nExpected: Both should produce the same hash values")
print("Actual:   Different hashes for the same input!")
print("  '' hashes to:    ", hash_without_categorize[0])
print("  '\\x00' hashes to:", hash_without_categorize[1])
print("\nBut with categorize=True, both hash to:", hash_with_categorize[0])

# Additional test to verify if they are indeed the same
print("\n=== VERIFICATION ===")
print("Are the hashes equal with categorize=True?", np.array_equal(hash_with_categorize, hash_without_categorize))
print("Hash for '' with categorize=True:", hash_with_categorize[0])
print("Hash for '\\x00' with categorize=True:", hash_with_categorize[1])
print("Are they the same?", hash_with_categorize[0] == hash_with_categorize[1])

# Let's also check factorize directly to see what's happening
from pandas import factorize
codes, categories = factorize(arr, sort=False)
print("\n=== FACTORIZE RESULTS ===")
print("Codes:", codes)
print("Categories:", categories)
print("Number of unique categories:", len(categories))