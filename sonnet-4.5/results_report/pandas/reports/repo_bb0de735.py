from pandas.io.sas.sas7bdat import _parse_datetime
from datetime import datetime

# Test the exact failing input from the bug report
sas_datetime = 2936550.0

try:
    result = _parse_datetime(sas_datetime, unit='d')
    print(f"Result: {result}")
except OverflowError as e:
    print(f"OverflowError: {e}")

# Also test the boundary case
print("\nBoundary testing:")
print("Testing 2936549.0 (should work):")
try:
    result = _parse_datetime(2936549.0, unit='d')
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {e}")

print("\nTesting 2936550.0 (should fail):")
try:
    result = _parse_datetime(2936550.0, unit='d')
    print(f"Result: {result}")
except Exception as e:
    print(f"Error type: {type(e).__name__}: {e}")