import numpy as np
from pandas.core.indexers import length_of_indexer

# Test case from the bug report
target = np.array([0])
slc = slice(None, None, -1)

result = length_of_indexer(slc, target)
actual = len(target[slc])

print(f"Test 1: Array with single element [0], slice(None, None, -1)")
print(f"  length_of_indexer returned: {result}")
print(f"  Actual length: {actual}")
print(f"  Match: {result == actual}")
print()

# Additional test cases from the report
test_cases = [
    (np.arange(5), slice(None, None, -1), "Array [0,1,2,3,4], slice(None, None, -1)"),
    (np.arange(10), slice(None, None, -2), "Array [0..9], slice(None, None, -2)"),
    (np.arange(10), slice(5, None, -1), "Array [0..9], slice(5, None, -1)"),
]

for i, (arr, slc, desc) in enumerate(test_cases, 2):
    result = length_of_indexer(slc, arr)
    actual = len(arr[slc])
    print(f"Test {i}: {desc}")
    print(f"  length_of_indexer returned: {result}")
    print(f"  Actual length: {actual}")
    print(f"  Match: {result == actual}")
    print()

# The assertion that fails
print("Running assertion from bug report:")
try:
    target = np.array([0])
    slc = slice(None, None, -1)
    result = length_of_indexer(slc, target)
    actual = len(target[slc])
    assert result == actual, f"Expected {actual}, got {result}"
    print("Assertion passed")
except AssertionError as e:
    print(f"AssertionError: {e}")