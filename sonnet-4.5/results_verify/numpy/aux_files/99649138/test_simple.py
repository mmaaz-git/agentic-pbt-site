import numpy.ma as ma
import traceback

# Test from the bug report
mask1 = [False, True, False]
mask2 = [True, False, False]

print("Testing with lists:")
print(f"mask1 = {mask1}")
print(f"mask2 = {mask2}")

try:
    result = ma.mask_or(mask1, mask2)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    print("\nFull traceback:")
    traceback.print_exc()

# Test with numpy arrays for comparison
print("\n" + "="*50)
print("Testing with numpy arrays:")
mask1_arr = ma.array(mask1)
mask2_arr = ma.array(mask2)
print(f"mask1_arr = {mask1_arr}")
print(f"mask2_arr = {mask2_arr}")

try:
    result = ma.mask_or(mask1_arr, mask2_arr)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    traceback.print_exc()

# Test with boolean numpy arrays
print("\n" + "="*50)
print("Testing with boolean numpy arrays:")
mask1_bool = ma.array([False, True, False], dtype=bool)
mask2_bool = ma.array([True, False, False], dtype=bool)
print(f"mask1_bool = {mask1_bool}")
print(f"mask2_bool = {mask2_bool}")

try:
    result = ma.mask_or(mask1_bool, mask2_bool)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    traceback.print_exc()