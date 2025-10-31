import numpy as np
from pandas.core.internals.base import ensure_np_dtype

# Test with variable-length Unicode dtype (returned by np.dtype(str))
dtype_var_length = np.dtype(str)
print(f"Variable-length dtype: {dtype_var_length}")
result_var = ensure_np_dtype(dtype_var_length)
print(f"Result after ensure_np_dtype: {result_var}")
print(f"Converted to object? {result_var == np.dtype('object')}")
print()

# Test with fixed-length Unicode dtype
dtype_fixed_10 = np.dtype('U10')
print(f"Fixed-length dtype (U10): {dtype_fixed_10}")
result_fixed_10 = ensure_np_dtype(dtype_fixed_10)
print(f"Result after ensure_np_dtype: {result_fixed_10}")
print(f"Converted to object? {result_fixed_10 == np.dtype('object')}")
print()

# Test with another fixed-length Unicode dtype
dtype_fixed_100 = np.dtype('U100')
print(f"Fixed-length dtype (U100): {dtype_fixed_100}")
result_fixed_100 = ensure_np_dtype(dtype_fixed_100)
print(f"Result after ensure_np_dtype: {result_fixed_100}")
print(f"Converted to object? {result_fixed_100 == np.dtype('object')}")
print()

# Show that both types have the same 'kind'
print(f"Variable-length dtype.kind: {dtype_var_length.kind}")
print(f"Fixed-length (U10) dtype.kind: {dtype_fixed_10.kind}")
print(f"Fixed-length (U100) dtype.kind: {dtype_fixed_100.kind}")
print()

# This assertion will pass
assert result_var == np.dtype('object'), "Variable-length Unicode should convert to object"

# This assertion will fail, demonstrating the bug
try:
    assert result_fixed_10 == np.dtype('object'), "Fixed-length Unicode should also convert to object"
    print("BUG NOT PRESENT: Fixed-length Unicode was correctly converted to object")
except AssertionError as e:
    print(f"BUG CONFIRMED: {e}")