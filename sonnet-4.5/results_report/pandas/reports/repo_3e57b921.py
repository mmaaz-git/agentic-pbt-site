import numpy as np
from pandas.core.indexers import length_of_indexer

target = np.arange(50)

# Test case 1: slice(1, 0, None) - start > stop with positive step
slc1 = slice(1, 0, None)
print(f"slice(1, 0, None):")
print(f"  length_of_indexer: {length_of_indexer(slc1, target)}")
print(f"  Actual length: {len(target[slc1])}")
print(f"  Actual sliced array: {target[slc1]}")

# Test case 2: slice(0, 1, -1) - start < stop with negative step
slc2 = slice(0, 1, -1)
print(f"\nslice(0, 1, -1):")
print(f"  length_of_indexer: {length_of_indexer(slc2, target)}")
print(f"  Actual length: {len(target[slc2])}")
print(f"  Actual sliced array: {target[slc2]}")

# Test case 3: slice(None, None, -1) - full negative slice
slc3 = slice(None, None, -1)
print(f"\nslice(None, None, -1):")
print(f"  length_of_indexer: {length_of_indexer(slc3, target)}")
print(f"  Actual length: {len(target[slc3])}")

# Test case 4: slice(-1, 9, None) - negative start index
slc4 = slice(-1, 9, None)
print(f"\nslice(-1, 9, None):")
print(f"  length_of_indexer: {length_of_indexer(slc4, target)}")
print(f"  Actual length: {len(target[slc4])}")

# Test case 5: slice(10, 5, None) - another start > stop case
slc5 = slice(10, 5, None)
print(f"\nslice(10, 5, None):")
print(f"  length_of_indexer: {length_of_indexer(slc5, target)}")
print(f"  Actual length: {len(target[slc5])}")
print(f"  Actual sliced array: {target[slc5]}")