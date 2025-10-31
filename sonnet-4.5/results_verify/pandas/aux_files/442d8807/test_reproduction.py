import numpy as np
from pandas.core.indexers import length_of_indexer

array = np.array([0])
indexer = slice(2, None, None)

actual_result = array[indexer]
actual_length = len(actual_result)
predicted_length = length_of_indexer(indexer, array)

print(f"Array: {array}")
print(f"Indexer: {indexer}")
print(f"Actual result: {actual_result}")
print(f"Actual length: {actual_length}")
print(f"Predicted length: {predicted_length}")
print(f"Bug: {predicted_length < 0}")

# Test a few more cases
print("\nAdditional test cases:")

# Case 1: Start > length with explicit stop
array2 = np.array([1, 2, 3])
indexer2 = slice(5, 10, None)
actual_length2 = len(array2[indexer2])
predicted_length2 = length_of_indexer(indexer2, array2)
print(f"Array length: {len(array2)}, slice(5, 10, None)")
print(f"  Actual length: {actual_length2}, Predicted: {predicted_length2}, Negative: {predicted_length2 < 0}")

# Case 2: Negative stop that goes before start
array3 = np.array([1])
indexer3 = slice(None, -2, None)
actual_length3 = len(array3[indexer3])
predicted_length3 = length_of_indexer(indexer3, array3)
print(f"Array length: {len(array3)}, slice(None, -2, None)")
print(f"  Actual length: {actual_length3}, Predicted: {predicted_length3}, Negative: {predicted_length3 < 0}")

# Case 3: Start > stop after normalization
array4 = np.array([1, 2, 3, 4, 5])
indexer4 = slice(3, 2, None)
actual_length4 = len(array4[indexer4])
predicted_length4 = length_of_indexer(indexer4, array4)
print(f"Array length: {len(array4)}, slice(3, 2, None)")
print(f"  Actual length: {actual_length4}, Predicted: {predicted_length4}, Negative: {predicted_length4 < 0}")