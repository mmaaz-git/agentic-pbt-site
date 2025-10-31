import numpy as np
from pandas.core.indexers import length_of_indexer

# Test case from the bug report
target = np.array([0])
slc = slice(1, 0, 1)

computed_length = length_of_indexer(slc, target)
actual_length = len(target[slc])

print(f"Test case 1: slice(1, 0, 1) on array of length 1")
print(f"Computed length: {computed_length}")
print(f"Actual length: {actual_length}")
print(f"Match: {computed_length == actual_length}")
print()

# Additional test cases
test_cases = [
    (slice(5, 3, 1), np.arange(10)),
    (slice(10, 5, 1), np.arange(20)),
    (slice(3, 5, -1), np.arange(10)),
    (slice(2, 1, 1), np.arange(5)),
    (slice(100, 50, 1), np.arange(200))
]

for i, (slc, target) in enumerate(test_cases, 2):
    computed_length = length_of_indexer(slc, target)
    actual_length = len(target[slc])

    print(f"Test case {i}: {slc} on array of length {len(target)}")
    print(f"Computed length: {computed_length}")
    print(f"Actual length: {actual_length}")
    print(f"Match: {computed_length == actual_length}")
    print()

# Demonstrate the assertion error
print("Assertion error on the original test case:")
target = np.array([0])
slc = slice(1, 0, 1)
computed_length = length_of_indexer(slc, target)
actual_length = len(target[slc])
assert computed_length == actual_length, f"Computed {computed_length} != Actual {actual_length}"