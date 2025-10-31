import numpy as np
from pandas.api.extensions import take

# Test case from bug report
arr = np.array([0, 0, 0, 0, 0])
negative_val = -9_223_372_036_854_775_809

print(f"Array: {arr}")
print(f"Negative value: {negative_val}")
print(f"Negative value is beyond C long range: {negative_val < -2**63}")

try:
    result = take(arr, [0, negative_val], allow_fill=True)
    print(f"Result (no exception): {result}")
except ValueError as e:
    print(f"Got expected ValueError: {e}")
except OverflowError as e:
    print(f"BUG: Got OverflowError instead of ValueError: {e}")
except Exception as e:
    print(f"Got unexpected exception {type(e).__name__}: {e}")

# Additional test with a smaller negative value
print("\n--- Test with smaller negative value ---")
try:
    result = take(arr, [0, -2], allow_fill=True)
    print(f"Result (no exception): {result}")
except ValueError as e:
    print(f"Got expected ValueError: {e}")
except OverflowError as e:
    print(f"Got OverflowError: {e}")
except Exception as e:
    print(f"Got unexpected exception {type(e).__name__}: {e}")

# Test with -1 which should be allowed
print("\n--- Test with -1 (should work) ---")
try:
    result = take(arr, [0, -1], allow_fill=True)
    print(f"Result with -1: {result}")
except Exception as e:
    print(f"Got exception {type(e).__name__}: {e}")