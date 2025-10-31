import numpy as np
from pandas.core.indexers import length_of_indexer

# Bug 1: Negative step slices return negative lengths
print("Bug 1: Negative step slices")
target = np.array([0])
result = length_of_indexer(slice(None, None, -1), target)
actual = len(target[slice(None, None, -1)])
print(f"Target: {target}")
print(f"Slice: slice(None, None, -1)")
print(f"length_of_indexer result: {result}")
print(f"Actual length: {actual}")
print(f"Expected: {actual}, Got: {result}")
print()

# Bug 2: Range with step > (stop - start) returns wrong length
print("Bug 2: Range with large step")
r = range(0, 1, 2)
result = length_of_indexer(r, None)
actual = len(r)
print(f"Range: range(0, 1, 2)")
print(f"length_of_indexer result: {result}")
print(f"Actual length: {actual}")
print(f"Expected: {actual}, Got: {result}")
print()

# Bug 3: Slice with start > len(target) returns negative
print("Bug 3: Slice with start > len(target)")
target = np.array([0])
result = length_of_indexer(slice(2, None), target)
actual = len(target[slice(2, None)])
print(f"Target: {target}")
print(f"Slice: slice(2, None)")
print(f"length_of_indexer result: {result}")
print(f"Actual length: {actual}")
print(f"Expected: {actual}, Got: {result}")
print()

# Bug 4: Slice with negative stop out of bounds returns negative
print("Bug 4: Slice with negative stop out of bounds")
target = np.array([0])
result = length_of_indexer(slice(None, -5), target)
actual = len(target[slice(None, -5)])
print(f"Target: {target}")
print(f"Slice: slice(None, -5)")
print(f"length_of_indexer result: {result}")
print(f"Actual length: {actual}")
print(f"Expected: {actual}, Got: {result}")