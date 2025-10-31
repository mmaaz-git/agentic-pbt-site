import numpy as np
from pandas._libs.algos import unique_deltas

# Test with single element
single = np.array([5], dtype=np.int64)
print(f"Single element array: {single}")
print(f"unique_deltas result: {unique_deltas(single)}")
print(f"Length: {len(unique_deltas(single))}")

# Test with two elements
two = np.array([5, 10], dtype=np.int64)
print(f"\nTwo element array: {two}")
print(f"unique_deltas result: {unique_deltas(two)}")
print(f"Length: {len(unique_deltas(two))}")

# Test with three elements
three = np.array([5, 10, 15], dtype=np.int64)
print(f"\nThree element array: {three}")
print(f"unique_deltas result: {unique_deltas(three)}")
print(f"Length: {len(unique_deltas(three))}")

# Test empty array
empty = np.array([], dtype=np.int64)
print(f"\nEmpty array: {empty}")
print(f"unique_deltas result: {unique_deltas(empty)}")
print(f"Length: {len(unique_deltas(empty))}")