import numpy as np
from xarray.plot.utils import _rescale_imshow_rgb

# Test case 1: Single constant value
print("Test 1: Single constant value [0.0]")
darray = np.array([0.0])
result = _rescale_imshow_rgb(darray, vmin=None, vmax=None, robust=True)
print(f"Input: {darray}")
print(f"Result: {result}")
print(f"Contains NaN: {np.any(np.isnan(result))}")
print()

# Test case 2: Multiple identical values
print("Test 2: Multiple identical values [5.0, 5.0, 5.0]")
darray2 = np.array([5.0, 5.0, 5.0])
result2 = _rescale_imshow_rgb(darray2, vmin=None, vmax=None, robust=True)
print(f"Input: {darray2}")
print(f"Result: {result2}")
print(f"Contains NaN: {np.any(np.isnan(result2))}")
print()

# Test case 3: Explicitly set vmin=vmax
print("Test 3: Explicitly set vmin=vmax with [1.0, 2.0, 3.0]")
darray3 = np.array([1.0, 2.0, 3.0])
result3 = _rescale_imshow_rgb(darray3, vmin=2.0, vmax=2.0, robust=False)
print(f"Input: {darray3}")
print(f"Result: {result3}")
print(f"Contains NaN: {np.any(np.isnan(result3))}")