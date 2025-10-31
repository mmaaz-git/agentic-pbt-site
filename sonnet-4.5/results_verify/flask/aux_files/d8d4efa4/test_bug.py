import numpy as np
from pandas.core.array_algos.masked_reductions import sum as masked_sum

print("Testing the bug reproduction...")

values_obj = np.array([[1, 2], [3, 4]], dtype=object)
mask = np.array([[True, False], [False, False]], dtype=bool)

print(f"values_obj:\n{values_obj}")
print(f"mask:\n{mask}")

try:
    result = masked_sum(values_obj, mask, skipna=True, axis=1)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\nNow testing with float dtype for comparison:")
values_float = values_obj.astype(float)
try:
    result_float = masked_sum(values_float, mask, skipna=True, axis=1)
    print(f"Float result: {result_float}")
except Exception as e:
    print(f"Float error: {type(e).__name__}: {e}")