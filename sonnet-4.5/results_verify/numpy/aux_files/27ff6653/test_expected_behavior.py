import numpy.ma as ma
import numpy as np

# Test what happens with numpy arrays (the expected working case)
print("Testing with numpy arrays:")
m1_array = np.array([False])
m2_array = np.array([False])
try:
    result = ma.mask_or(m1_array, m2_array)
    print(f"Success with numpy arrays: {result}")
except Exception as e:
    print(f"Failed with numpy arrays: {e}")

# Test that make_mask works with lists
print("\nTesting make_mask with lists (as shown in examples):")
try:
    mask1 = ma.make_mask([0, 1, 1, 0])
    mask2 = ma.make_mask([1, 0, 0, 0])
    print(f"make_mask with lists works: mask1={mask1}, mask2={mask2}")
    result = ma.mask_or(mask1, mask2)
    print(f"mask_or with make_mask results: {result}")
except Exception as e:
    print(f"Failed: {e}")

# Test behavior with plain lists
print("\nTesting with plain lists directly:")
try:
    result = ma.mask_or([False], [False])
    print(f"Success with plain lists: {result}")
except Exception as e:
    print(f"Failed with plain lists: {e}")

# Test that the underlying logical_or works with lists
print("\nTesting numpy logical_or with lists:")
try:
    result = np.logical_or([False], [False])
    print(f"np.logical_or works with lists: {result}")
except Exception as e:
    print(f"Failed: {e}")