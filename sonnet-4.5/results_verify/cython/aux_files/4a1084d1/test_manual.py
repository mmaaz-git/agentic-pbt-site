import numpy as np
from pandas.core.indexers import length_of_indexer

# Test the specific case in the bug report
print("Testing the specific case from the bug report:")
target = np.array([0])
slc = slice(None, None, -1)

result = length_of_indexer(slc, target)
actual = len(target[slc])

print(f"length_of_indexer: {result}")
print(f"Actual length: {actual}")
print(f"Match: {result == actual}")
print()

# Test additional cases mentioned in the bug report
print("Testing additional cases:")

# Case 1: slice(None, None, -1) on array of length 5
target = np.arange(5)
slc = slice(None, None, -1)
result = length_of_indexer(slc, target)
actual = len(target[slc])
print(f"Array length 5, slice(None, None, -1):")
print(f"  length_of_indexer: {result}, actual: {actual}, match: {result == actual}")

# Case 2: slice(None, None, -2) on array of length 10
target = np.arange(10)
slc = slice(None, None, -2)
result = length_of_indexer(slc, target)
actual = len(target[slc])
print(f"Array length 10, slice(None, None, -2):")
print(f"  length_of_indexer: {result}, actual: {actual}, match: {result == actual}")

# Case 3: slice(5, None, -1) on array of length 10
target = np.arange(10)
slc = slice(5, None, -1)
result = length_of_indexer(slc, target)
actual = len(target[slc])
print(f"Array length 10, slice(5, None, -1):")
print(f"  length_of_indexer: {result}, actual: {actual}, match: {result == actual}")