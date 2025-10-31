import numpy as np
from pandas.core.indexers import length_of_indexer

print("Testing the bug cases:")
print("-" * 40)

# Case 1: Empty array, start=1
target = np.arange(0)
slc = slice(1, None, None)
computed_length = length_of_indexer(slc, target)
actual_length = len(target[slc])
print(f"Case 1 - Empty array, slice(1, None):")
print(f"  Computed: {computed_length}, Actual: {actual_length}")
print(f"  Match? {computed_length == actual_length}")
print()

# Case 2: Array of 5, start=10
target = np.arange(5)
slc = slice(10, None, None)
computed_length = length_of_indexer(slc, target)
actual_length = len(target[slc])
print(f"Case 2 - Array[0-4], slice(10, None):")
print(f"  Computed: {computed_length}, Actual: {actual_length}")
print(f"  Match? {computed_length == actual_length}")
print()

# Let's test more cases
print("Additional test cases:")
print("-" * 40)

# Case 3: Array of 3, start=3 (exactly at boundary)
target = np.arange(3)
slc = slice(3, None, None)
computed_length = length_of_indexer(slc, target)
actual_length = len(target[slc])
print(f"Case 3 - Array[0-2], slice(3, None):")
print(f"  Computed: {computed_length}, Actual: {actual_length}")
print(f"  Match? {computed_length == actual_length}")
print()

# Case 4: Array of 5, start=2, stop=10
target = np.arange(5)
slc = slice(2, 10, None)
computed_length = length_of_indexer(slc, target)
actual_length = len(target[slc])
print(f"Case 4 - Array[0-4], slice(2, 10):")
print(f"  Computed: {computed_length}, Actual: {actual_length}")
print(f"  Match? {computed_length == actual_length}")
print()