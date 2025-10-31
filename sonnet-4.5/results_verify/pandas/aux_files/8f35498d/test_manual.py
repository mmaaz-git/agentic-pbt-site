import numpy as np
from pandas.core.indexers import length_of_indexer

target = np.array([0])

indexer1 = slice(None, -2, None)
print(f"Case 1: {indexer1}")
print(f"  Actual: {len(target[indexer1])}, Predicted: {length_of_indexer(indexer1, target)}")

indexer2 = slice(2, None)
print(f"Case 2: {indexer2}")
print(f"  Actual: {len(target[indexer2])}, Predicted: {length_of_indexer(indexer2, target)}")

# Additional test cases
indexer3 = slice(None, -10, None)
print(f"\nCase 3: {indexer3}")
print(f"  Actual: {len(target[indexer3])}, Predicted: {length_of_indexer(indexer3, target)}")

indexer4 = slice(5, None)
print(f"\nCase 4: {indexer4}")
print(f"  Actual: {len(target[indexer4])}, Predicted: {length_of_indexer(indexer4, target)}")

# Test with larger array
target2 = np.array([0, 1, 2, 3, 4])
indexer5 = slice(None, -10, None)
print(f"\nCase 5 (larger array): {indexer5}")
print(f"  Actual: {len(target2[indexer5])}, Predicted: {length_of_indexer(indexer5, target2)}")

indexer6 = slice(10, None)
print(f"\nCase 6 (larger array): {indexer6}")
print(f"  Actual: {len(target2[indexer6])}, Predicted: {length_of_indexer(indexer6, target2)}")