import numpy as np
from xarray.plot.utils import _rescale_imshow_rgb

# Test case 1: vmin > vmax (should raise ValueError but doesn't)
darray = np.array([[[50.0, 50.0, 50.0]]]).astype('f8')

print("Test 1: vmin=100.0, vmax=0.0")
print("Expected: ValueError")
print("Actual:")
try:
    result = _rescale_imshow_rgb(darray, vmin=100.0, vmax=0.0, robust=False)
    print(f"No error raised! Result: {result}")
except ValueError as e:
    print(f"ValueError raised: {e}")

print("\n" + "="*50 + "\n")

# Test case 2: vmin == vmax (should raise ValueError but causes division by zero)
print("Test 2: vmin=50.0, vmax=50.0")
print("Expected: ValueError")
print("Actual:")
try:
    result = _rescale_imshow_rgb(darray, vmin=50.0, vmax=50.0, robust=False)
    print(f"No error raised! Result: {result}")
except ValueError as e:
    print(f"ValueError raised: {e}")

print("\n" + "="*50 + "\n")

# Test case 3: For comparison - existing validation works (vmax=None, vmin too high)
print("Test 3: vmin=500.0, vmax=None (existing validation)")
print("Expected: ValueError")
print("Actual:")
try:
    result = _rescale_imshow_rgb(darray, vmin=500.0, vmax=None, robust=False)
    print(f"No error raised! Result: {result}")
except ValueError as e:
    print(f"ValueError raised: {e}")