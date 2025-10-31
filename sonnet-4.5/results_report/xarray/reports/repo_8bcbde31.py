import numpy as np
from xarray.core import duck_array_ops

data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
where = np.array([True, False, True, False, True])

numpy_result = np.sum(data, where=where)
xarray_result = duck_array_ops.sum_where(data, where=where)

print(f"Data array: {data}")
print(f"Where mask: {where}")
print(f"numpy.sum(data, where=where): {numpy_result}")
print(f"xarray sum_where(data, where=where): {xarray_result}")
print(f"Expected sum (1.0 + 3.0 + 5.0): 9.0")
print(f"Actual xarray sum (2.0 + 4.0): {xarray_result}")
print(f"\nAre results equal? {numpy_result == xarray_result}")

assert numpy_result != xarray_result, "Bug confirmed: sum_where has inverted logic"
print("\nBug confirmed: sum_where sums where condition is False instead of True")