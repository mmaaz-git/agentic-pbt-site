import numpy as np
from pandas.core.ops.common import _maybe_match_name


class MockObj:
    def __init__(self, name):
        self.name = name


arr1 = np.array([1, 2, 3])
arr2 = np.array([1, 2, 3])

a = MockObj(arr1)
b = MockObj(arr2)

result = _maybe_match_name(a, b)

print(f"Result: {result}")
print(f"Expected: {arr1}")
print(f"Arrays are equal: {np.array_equal(arr1, arr2)}")