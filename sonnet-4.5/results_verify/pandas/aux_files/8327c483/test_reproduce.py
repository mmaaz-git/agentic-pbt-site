import pandas as pd
import numpy as np

print("Testing pandas.qcut with tiny float values...")

# Test case from the bug report
arr = [0.0] * 19 + [5e-324]

print(f"\nInput array: {arr}")
print(f"Unique values in array: {list(set(arr))}")

# Test with duplicates='drop'
print("\n1. Testing with duplicates='drop':")
try:
    result = pd.qcut(arr, q=2, duplicates='drop')
    print(f"Success: {result}")
    print(f"Value counts: {result.value_counts()}")
except ValueError as e:
    print(f"ValueError with duplicates='drop': {e}")
except Exception as e:
    print(f"Other error with duplicates='drop': {type(e).__name__}: {e}")

# Test without duplicates='drop' (default)
print("\n2. Testing without duplicates='drop' (default):")
try:
    result = pd.qcut(arr, q=2)
    print(f"Success: {result}")
    print(f"Value counts: {result.value_counts()}")
except ValueError as e:
    print(f"ValueError without duplicates='drop': {e}")
except Exception as e:
    print(f"Other error without duplicates='drop': {type(e).__name__}: {e}")

# Let's also test what the _round_frac function does with tiny values
print("\n3. Testing _round_frac behavior on tiny floats:")
from pandas.core.reshape.tile import _round_frac

test_values = [0.0, 5e-324, 1e-300, 1e-100, 1e-10, 0.1, 1.0]
for val in test_values:
    try:
        rounded = _round_frac(val, 3)
        print(f"_round_frac({val}, 3) = {rounded}")
    except Exception as e:
        print(f"_round_frac({val}, 3) raised {type(e).__name__}: {e}")

# Check what np.around does with large digits
print("\n4. Testing np.around with large digits values:")
for digits in [10, 100, 200, 300, 323, 324]:
    result = np.around(5e-324, digits)
    print(f"np.around(5e-324, {digits}) = {result} (is NaN: {np.isnan(result)})")