import numpy.ma as ma
import traceback

print("Testing simple reproduction case:")
m1 = [False]
m2 = [False]

try:
    result = ma.mask_or(m1, m2)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")
    traceback.print_exc()