import numpy as np
from pandas.core.indexers import length_of_indexer

arr = np.arange(0)

print(f"slice(None, -1, None): expected 0, got {length_of_indexer(slice(None, -1, None), arr)}")
print(f"slice(None, -2, None): expected 0, got {length_of_indexer(slice(None, -2, None), arr)}")
print(f"slice(-1, None, None): expected 0, got {length_of_indexer(slice(-1, None, None), arr)}")
print(f"slice(0, -1, None): expected 0, got {length_of_indexer(slice(0, -1, None), arr)}")

# Let's also verify what actual slicing returns
print("\nActual slicing behavior:")
print(f"arr[slice(None, -1, None)]: {arr[slice(None, -1, None)]}, len={len(arr[slice(None, -1, None)])}")
print(f"arr[slice(None, -2, None)]: {arr[slice(None, -2, None)]}, len={len(arr[slice(None, -2, None)])}")
print(f"arr[slice(-1, None, None)]: {arr[slice(-1, None, None)]}, len={len(arr[slice(-1, None, None)])}")
print(f"arr[slice(0, -1, None)]: {arr[slice(0, -1, None)]}, len={len(arr[slice(0, -1, None)])}")