import numpy as np
from xarray.core import duck_array_ops

arr = np.array([[1, 2], [3, 4]])

xarray_cumprod = duck_array_ops.cumprod(arr, axis=None)
numpy_cumprod = np.cumprod(arr, axis=None)

print(f"xarray result: shape={xarray_cumprod.shape}, values={xarray_cumprod.flatten().tolist()}")
print(f"numpy result:  shape={numpy_cumprod.shape}, values={numpy_cumprod.tolist()}")

xarray_cumsum = duck_array_ops.cumsum(arr, axis=None)
numpy_cumsum = np.cumsum(arr, axis=None)

print(f"\nxarray cumsum: shape={xarray_cumsum.shape}, values={xarray_cumsum.flatten().tolist()}")
print(f"numpy cumsum:  shape={numpy_cumsum.shape}, values={numpy_cumsum.tolist()}")