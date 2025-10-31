import numpy as np
from pandas.core.indexers import length_of_indexer

# Test case 1: slice(1, 0, 1)
target = np.arange(5)
indexer = slice(1, 0, 1)
actual_length = len(target[indexer])
predicted_length = length_of_indexer(indexer, target)
print(f"Test 1 - slice(1, 0, 1) on array of len 5:")
print(f"  Actual: {actual_length}, Predicted: {predicted_length}")

# Test case 2: slice(None, None, -1)
target = np.arange(10)
indexer = slice(None, None, -1)
actual_length = len(target[indexer])
predicted_length = length_of_indexer(indexer, target)
print(f"Test 2 - slice(None, None, -1) on array of len 10:")
print(f"  Actual: {actual_length}, Predicted: {predicted_length}")

# Additional test cases mentioned in bug report
# Test case 3: negative indices
target = np.arange(10)
indexer = slice(10, -11, 1)
actual_length = len(target[indexer])
predicted_length = length_of_indexer(indexer, target)
print(f"Test 3 - slice(10, -11, 1) on array of len 10:")
print(f"  Actual: {actual_length}, Predicted: {predicted_length}")

# Test case 4: start beyond bounds
target = np.arange(1)
indexer = slice(5, 101, 1)
actual_length = len(target[indexer])
predicted_length = length_of_indexer(indexer, target)
print(f"Test 4 - slice(5, 101, 1) on array of len 1:")
print(f"  Actual: {actual_length}, Predicted: {predicted_length}")