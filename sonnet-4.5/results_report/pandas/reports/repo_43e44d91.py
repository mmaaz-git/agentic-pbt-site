import numpy as np
from pandas.core.indexers import length_of_indexer

# Test case from the bug report
target = np.arange(0)
slc = slice(1, None, None)

computed_length = length_of_indexer(slc, target)
actual_length = len(target[slc])

print(f"Computed length: {computed_length}")
print(f"Actual length: {actual_length}")
print(f"Bug: {computed_length} != {actual_length}")
print()

# Additional test cases
test_cases = [
    (0, slice(1, None), "Empty array, start=1"),
    (0, slice(5, 10), "Empty array, start=5, stop=10"),
    (3, slice(5, None), "Array[3], start=5"),
    (5, slice(3, 2), "Array[5], start=3, stop=2"),
]

print("Additional test cases:")
for target_len, slc, desc in test_cases:
    target = np.arange(target_len)
    computed = length_of_indexer(slc, target)
    actual = len(target[slc])
    print(f"{desc}: computed={computed}, actual={actual}, match={computed == actual}")