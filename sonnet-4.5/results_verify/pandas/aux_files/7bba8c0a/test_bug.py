import numpy as np
import pandas.util

# Test case from the bug report
arr = np.array(['', '', '\x00'], dtype=object)

hash_cat = pandas.util.hash_array(arr, categorize=True)
hash_no_cat = pandas.util.hash_array(arr, categorize=False)

print("Input array:", repr(arr))
print(f"categorize=True:  {hash_cat}")
print(f"categorize=False: {hash_no_cat}")

# Check if the hashes are different
print("\nAnalysis:")
print(f"Empty string ('') hashes with categorize=True: {hash_cat[0]}, {hash_cat[1]}")
print(f"Null byte ('\\x00') hash with categorize=True: {hash_cat[2]}")
print(f"Are all hashes same with categorize=True? {len(set(hash_cat)) == 1}")

print(f"\nEmpty string ('') hashes with categorize=False: {hash_no_cat[0]}, {hash_no_cat[1]}")
print(f"Null byte ('\\x00') hash with categorize=False: {hash_no_cat[2]}")
print(f"Are empty string hashes same with categorize=False? {hash_no_cat[0] == hash_no_cat[1]}")
print(f"Is null byte hash different with categorize=False? {hash_no_cat[2] != hash_no_cat[0]}")