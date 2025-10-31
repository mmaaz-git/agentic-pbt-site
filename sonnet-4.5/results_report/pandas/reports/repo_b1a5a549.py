from pandas.io.sas.sas7bdat import _parse_datetime

# Test with large value that causes OverflowError
large_value = 1e15

try:
    print(f"Testing _parse_datetime with value {large_value} (seconds)")
    result = _parse_datetime(large_value, 's')
    print(f"Result: {result}")
except OverflowError as e:
    print(f"OverflowError: {e}")

print()

try:
    print(f"Testing _parse_datetime with value {large_value} (days)")
    result = _parse_datetime(large_value, 'd')
    print(f"Result: {result}")
except OverflowError as e:
    print(f"OverflowError: {e}")

print()

# Test with NaN to show inconsistent handling
import numpy as np
print("Testing _parse_datetime with NaN value")
result = _parse_datetime(np.nan, 's')
print(f"NaN handling result: {result}")

print()

# Test with normal values that work
normal_value = 100000
print(f"Testing _parse_datetime with normal value {normal_value} (seconds)")
result = _parse_datetime(normal_value, 's')
print(f"Result: {result}")