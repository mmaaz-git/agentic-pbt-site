import numpy as np
import pandas.util

# The specific failing case
print("Testing the specific case: ['', '', '\\x00']")
arr = np.array(['', '', '\x00'], dtype=object)

hash_cat = pandas.util.hash_array(arr, categorize=True)
hash_no_cat = pandas.util.hash_array(arr, categorize=False)

print(f"\ncategorize=True:  {hash_cat}")
print(f"categorize=False: {hash_no_cat}")

# Now test what happens with just two items
print("\n\nTesting: ['', '\\x00'] (without duplicate)")
arr2 = np.array(['', '\x00'], dtype=object)
hash_cat2 = pandas.util.hash_array(arr2, categorize=True)
hash_no_cat2 = pandas.util.hash_array(arr2, categorize=False)

print(f"categorize=True:  {hash_cat2}")
print(f"categorize=False: {hash_no_cat2}")

# Test with duplicate null bytes
print("\n\nTesting: ['\\x00', '\\x00', ''] (duplicate null bytes)")
arr3 = np.array(['\x00', '\x00', ''], dtype=object)
hash_cat3 = pandas.util.hash_array(arr3, categorize=True)
hash_no_cat3 = pandas.util.hash_array(arr3, categorize=False)

print(f"categorize=True:  {hash_cat3}")
print(f"categorize=False: {hash_no_cat3}")

# Test with more empty strings
print("\n\nTesting: ['', '', '', '\\x00'] (three empty strings)")
arr4 = np.array(['', '', '', '\x00'], dtype=object)
hash_cat4 = pandas.util.hash_array(arr4, categorize=True)
hash_no_cat4 = pandas.util.hash_array(arr4, categorize=False)

print(f"categorize=True:  {hash_cat4}")
print(f"categorize=False: {hash_no_cat4}")