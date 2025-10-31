import numpy as np
import pandas.core.array_algos.quantile as quantile_module

# Test case from the bug report
values = np.array([-1, 127], dtype=np.int8)
qs = np.array([0.0, 0.5, 1.0])

result = quantile_module.quantile_compat(values, qs, 'linear')

print(f"Input array: {values}")
print(f"Input dtype: {values.dtype}")
print(f"Quantiles requested: {qs}")
print(f"Interpolation method: linear")
print()
print(f"Result: {result}")
print(f"Result dtype: {result.dtype}")
print()
print(f"Expected values (correct quantiles):")
print(f"  Q(0.0) should be: -1.0 (minimum)")
print(f"  Q(0.5) should be: 63.0 (median of -1 and 127)")
print(f"  Q(1.0) should be: 127.0 (maximum)")
print()
print(f"Actual values:")
print(f"  Q(0.0) = {result[0]}")
print(f"  Q(0.5) = {result[1]}")
print(f"  Q(1.0) = {result[2]}")
print()
print(f"Is monotonic (Q(0) <= Q(0.5) <= Q(1))? {result[0] <= result[1] <= result[2]}")
print(f"Median > Maximum? {result[1] > result[2]} (This should be False!)")