import numpy as np
from xarray.core import duck_array_ops

# Test array
arr = np.array([[1, 2], [3, 4]])

print("Testing cumprod with axis=None")
print("=" * 50)

# xarray cumprod
xarray_cumprod = duck_array_ops.cumprod(arr, axis=None)
print(f"xarray result: shape={xarray_cumprod.shape}, values={xarray_cumprod.flatten().tolist()}")

# numpy cumprod
numpy_cumprod = np.cumprod(arr, axis=None)
print(f"numpy result:  shape={numpy_cumprod.shape}, values={numpy_cumprod.tolist()}")

print("\nTesting cumsum with axis=None")
print("=" * 50)

# xarray cumsum
xarray_cumsum = duck_array_ops.cumsum(arr, axis=None)
print(f"xarray cumsum: shape={xarray_cumsum.shape}, values={xarray_cumsum.flatten().tolist()}")

# numpy cumsum
numpy_cumsum = np.cumsum(arr, axis=None)
print(f"numpy cumsum:  shape={numpy_cumsum.shape}, values={numpy_cumsum.tolist()}")