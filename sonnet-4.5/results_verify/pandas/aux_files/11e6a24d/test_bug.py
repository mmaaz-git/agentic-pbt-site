import numpy as np
from pandas.api.extensions import take

# Test 1: Reproduce the exact bug from the report
print("Test 1: Reproducing the bug with extremely large negative index")
arr = np.array([1])
idx = -9_223_372_036_854_775_809

try:
    result = take(arr, [idx])
    print(f"No error raised (unexpected)")
except OverflowError as e:
    print(f"OverflowError raised (BUG according to report): {e}")
except IndexError as e:
    print(f"IndexError raised (expected): {e}")

print("\n" + "="*50 + "\n")

# Test 2: Regular out of bounds index
print("Test 2: Regular out of bounds negative index")
arr = np.array([1, 2, 3])
idx = -10  # Out of bounds for array of size 3

try:
    result = take(arr, [idx])
    print(f"No error raised (unexpected)")
except OverflowError as e:
    print(f"OverflowError raised: {e}")
except IndexError as e:
    print(f"IndexError raised (expected): {e}")

print("\n" + "="*50 + "\n")

# Test 3: Check with positive out of bounds
print("Test 3: Positive out of bounds index")
arr = np.array([1, 2, 3])
idx = 10  # Out of bounds

try:
    result = take(arr, [idx])
    print(f"No error raised (unexpected)")
except OverflowError as e:
    print(f"OverflowError raised: {e}")
except IndexError as e:
    print(f"IndexError raised (expected): {e}")

print("\n" + "="*50 + "\n")

# Test 4: Very large positive index that might overflow
print("Test 4: Extremely large positive index")
arr = np.array([1])
idx = 9_223_372_036_854_775_807  # Maximum long value

try:
    result = take(arr, [idx])
    print(f"No error raised (unexpected)")
except OverflowError as e:
    print(f"OverflowError raised: {e}")
except IndexError as e:
    print(f"IndexError raised (expected): {e}")