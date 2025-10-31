import numpy as np
from pandas.core.indexers import length_of_indexer

target = np.arange(50)

# Test case 1: start > stop with positive step
slc1 = slice(1, 0, None)
print(f"slice(1, 0, None):")
print(f"  length_of_indexer: {length_of_indexer(slc1, target)}")
print(f"  Actual length: {len(target[slc1])}")
print(f"  Actual sliced array: {target[slc1]}")

# Test case 2: start < stop with negative step
slc2 = slice(0, 1, -1)
print(f"\nslice(0, 1, -1):")
print(f"  length_of_indexer: {length_of_indexer(slc2, target)}")
print(f"  Actual length: {len(target[slc2])}")
print(f"  Actual sliced array: {target[slc2]}")

# Additional test cases found from hypothesis
slc3 = slice(None, None, -1)
print(f"\nslice(None, None, -1):")
print(f"  length_of_indexer: {length_of_indexer(slc3, target)}")
print(f"  Actual length: {len(target[slc3])}")

slc4 = slice(0, -1, -2)
print(f"\nslice(0, -1, -2):")
print(f"  length_of_indexer: {length_of_indexer(slc4, target)}")
print(f"  Actual length: {len(target[slc4])}")
print(f"  Actual sliced array: {target[slc4]}")

slc5 = slice(-1, 9, None)
print(f"\nslice(-1, 9, None):")
print(f"  length_of_indexer: {length_of_indexer(slc5, target)}")
print(f"  Actual length: {len(target[slc5])}")
print(f"  Actual sliced array: {target[slc5]}")