import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import numpy as np
from xarray.plot.utils import _rescale_imshow_rgb

# Test with constant array of zeros
print("Test 1: Constant array of zeros")
darray = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
result = _rescale_imshow_rgb(darray, vmin=None, vmax=None, robust=True)
print(f"Input: {darray}")
print(f"Output: {result}")
print(f"Contains NaN: {np.any(np.isnan(result))}")
print()

# Test with constant array of fives
print("Test 2: Constant array of fives")
darray = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
result = _rescale_imshow_rgb(darray, vmin=None, vmax=None, robust=True)
print(f"Input: {darray}")
print(f"Output: {result}")
print(f"Contains NaN: {np.any(np.isnan(result))}")
print()

# Test with constant array of 255 (typical for image data)
print("Test 3: Constant array of 255 (white image)")
darray = np.array([255.0, 255.0, 255.0, 255.0, 255.0, 255.0])
result = _rescale_imshow_rgb(darray, vmin=None, vmax=None, robust=True)
print(f"Input: {darray}")
print(f"Output: {result}")
print(f"Contains NaN: {np.any(np.isnan(result))}")