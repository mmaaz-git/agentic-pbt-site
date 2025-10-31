import numpy as np
from pandas.core import roperator
from pandas.core.ops.array_ops import _masked_arith_op

# Test case 1: Basic rpow with base 1
x = np.array([0.0, 1.0, 2.0], dtype=object)
y = 1.0

result = _masked_arith_op(x, y, roperator.rpow)

print(f"Test 1: Basic rpow with base 1")
print(f"x = {x}")
print(f"y = {y}")
print(f"Result: {result}")
print(f"Expected: [1.0, 1.0, 1.0]")
print()

# Test case 2: rpow with base 1 and NaN values
x_with_nan = np.array([0.0, np.nan, 2.0], dtype=object)
y = 1.0

result_with_nan = _masked_arith_op(x_with_nan, y, roperator.rpow)

print(f"Test 2: rpow with base 1 and NaN values")
print(f"x = {x_with_nan}")
print(f"y = {y}")
print(f"Result: {result_with_nan}")
print(f"Expected: [1.0, nan, 1.0]")
print()

# Verify NumPy's behavior for comparison
print("NumPy's behavior for comparison:")
print(f"1.0 ** 0.0 = {1.0 ** 0.0}")
print(f"1.0 ** 1.0 = {1.0 ** 1.0}")
print(f"1.0 ** 2.0 = {1.0 ** 2.0}")
print(f"1.0 ** np.nan = {1.0 ** np.nan}")