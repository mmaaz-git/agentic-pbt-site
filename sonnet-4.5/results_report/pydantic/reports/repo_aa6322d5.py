import numpy as np
from pandas.core.util.hashing import hash_array
from pandas import factorize

# Test case 1: Basic hash collision demonstration
print("=" * 60)
print("Test 1: Basic Hash Collision with Empty String and Null Byte")
print("=" * 60)

data = ['', '\x00']
arr = np.array(data, dtype=object)

hash_with_categorize = hash_array(arr, categorize=True)
hash_without_categorize = hash_array(arr, categorize=False)

print(f"Input data: {repr(data)}")
print(f"Input array dtype: {arr.dtype}")
print()
print(f"Hash with categorize=True:  {hash_with_categorize}")
print(f"Hash with categorize=False: {hash_without_categorize}")
print()

# Check for hash collision
if hash_with_categorize[0] == hash_with_categorize[1]:
    print("❌ BUG CONFIRMED: Empty string '' and null byte '\\x00' have the SAME hash when categorize=True")
else:
    print("✓ No collision detected with categorize=True")

if hash_without_categorize[0] != hash_without_categorize[1]:
    print("✓ Correct: Empty string '' and null byte '\\x00' have DIFFERENT hashes when categorize=False")
else:
    print("❌ Unexpected: Hash collision even with categorize=False")

# Test case 2: Root cause analysis - factorize behavior
print("\n" + "=" * 60)
print("Test 2: Root Cause - factorize() Behavior")
print("=" * 60)

test_data = ['', '\x00', '\x00\x00', 'a', '\x00b']
codes, categories = factorize(test_data)

print(f"Input strings: {repr(test_data)}")
print(f"Factorize codes: {codes}")
print(f"Unique categories: {list(categories)}")
print()
print("Mapping:")
for i, val in enumerate(test_data):
    print(f"  {repr(val):10} -> code {codes[i]} -> category {repr(categories[codes[i]])}")

print()
if codes[0] == codes[1] == codes[2]:
    print("❌ BUG ROOT CAUSE: factorize() treats '', '\\x00', and '\\x00\\x00' as identical!")
    print("   All three distinct strings are mapped to the same category.")

# Test case 3: Impact demonstration
print("\n" + "=" * 60)
print("Test 3: Impact on Data Operations")
print("=" * 60)

# Show how this could affect real-world operations
distinct_values = ['', '\x00', 'a', 'b']
arr = np.array(distinct_values, dtype=object)

print(f"Distinct values: {repr(distinct_values)}")
print(f"Expected: 4 unique hash values")

hashes_categorized = hash_array(arr, categorize=True)
unique_hashes = len(set(hashes_categorized))

print(f"Actual unique hashes with categorize=True: {unique_hashes}")
print(f"Hash values: {hashes_categorized}")

if unique_hashes < len(distinct_values):
    print(f"❌ DATA INTEGRITY ISSUE: Only {unique_hashes} unique hashes for {len(distinct_values)} distinct values!")
    print("   This could cause incorrect groupby, deduplication, or equality checks.")