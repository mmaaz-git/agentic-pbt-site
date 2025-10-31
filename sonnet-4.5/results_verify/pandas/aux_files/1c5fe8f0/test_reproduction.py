import numpy as np
from pandas.api import extensions
import sys

print("Testing pandas.api.extensions.take with very large index")
print(f"sys.maxsize = {sys.maxsize}")

arr = np.array([0, 1, 2, 3, 4])
index_too_large = 9_223_372_036_854_775_808

print(f"\nArray: {arr}")
print(f"Index to test: {index_too_large}")
print(f"Index > sys.maxsize: {index_too_large > sys.maxsize}")

print("\nAttempting to call extensions.take(arr, [index_too_large])...")
try:
    result = extensions.take(arr, [index_too_large])
    print(f"Unexpected success! Result: {result}")
except OverflowError as e:
    print(f"OverflowError raised: {e}")
except IndexError as e:
    print(f"IndexError raised: {e}")
except Exception as e:
    print(f"Other exception raised: {type(e).__name__}: {e}")

print("\n--- Testing with normal out-of-bounds index for comparison ---")
normal_out_of_bounds = 100
print(f"Index to test: {normal_out_of_bounds}")

try:
    result = extensions.take(arr, [normal_out_of_bounds])
    print(f"Unexpected success! Result: {result}")
except OverflowError as e:
    print(f"OverflowError raised: {e}")
except IndexError as e:
    print(f"IndexError raised: {e}")
except Exception as e:
    print(f"Other exception raised: {type(e).__name__}: {e}")