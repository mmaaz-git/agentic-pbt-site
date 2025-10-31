from pandas.arrays import SparseArray
import numpy as np

left = SparseArray([1, 2, 3], fill_value=0)
right = SparseArray([1, 0, 1], fill_value=0)

result = left - right

print(f"result dense: {list(result.to_dense())}")
print(f"result.sp_values: {result.sp_values}")
print(f"result.fill_value: {result.fill_value}")
print(f"result.sp_index: {result.sp_index}")

# Check for the invariant violation
contains_fill = np.any(result.sp_values == result.fill_value)
print(f"\nInvariant violated (sp_values contains fill_value): {contains_fill}")