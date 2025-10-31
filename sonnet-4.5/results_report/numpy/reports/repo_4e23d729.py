import numpy.ma as ma

# Test case that should work according to documentation (array_like inputs)
mask1 = [False, True, False]
mask2 = [True, False, False]

print("Testing numpy.ma.mask_or with Python lists as inputs:")
print(f"mask1 = {mask1}")
print(f"mask2 = {mask2}")
print()

try:
    result = ma.mask_or(mask1, mask2)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()