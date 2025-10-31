from pandas.io.sas.sas7bdat import _parse_datetime

# Test with the failing input from the bug report
print("Testing with days:")
try:
    result = _parse_datetime(2936550.0, unit="d")
    print(f"Result: {result}")
except OverflowError as e:
    print(f"OverflowError: {e}")

print("\nTesting with seconds:")
try:
    result = _parse_datetime(1e15, unit="s")
    print(f"Result: {result}")
except OverflowError as e:
    print(f"OverflowError: {e}")