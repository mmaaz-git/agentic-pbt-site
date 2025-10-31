#!/usr/bin/env python3
"""Test the ensure_python_int bug report."""

from pandas.core.dtypes.common import ensure_python_int

# Test 1: Reproduce the exact bug from the report
print("Test 1: Testing ensure_python_int with float('inf')")
try:
    result = ensure_python_int(float('inf'))
    print(f"  Result: {result}")
except TypeError as e:
    print(f"  Correctly raised TypeError: {e}")
except OverflowError as e:
    print(f"  BUG: Raised OverflowError instead of TypeError: {e}")
except Exception as e:
    print(f"  Unexpected error: {type(e).__name__}: {e}")

print("\nTest 2: Testing ensure_python_int with float('-inf')")
try:
    result = ensure_python_int(float('-inf'))
    print(f"  Result: {result}")
except TypeError as e:
    print(f"  Correctly raised TypeError: {e}")
except OverflowError as e:
    print(f"  BUG: Raised OverflowError instead of TypeError: {e}")
except Exception as e:
    print(f"  Unexpected error: {type(e).__name__}: {e}")

print("\nTest 3: Testing ensure_python_int with NaN")
try:
    result = ensure_python_int(float('nan'))
    print(f"  Result: {result}")
except TypeError as e:
    print(f"  Correctly raised TypeError: {e}")
except ValueError as e:
    print(f"  Raised ValueError: {e}")
except Exception as e:
    print(f"  Unexpected error: {type(e).__name__}: {e}")

print("\nTest 4: Testing ensure_python_int with valid float")
try:
    result = ensure_python_int(5.0)
    print(f"  Result: {result} (type: {type(result)})")
except Exception as e:
    print(f"  Error: {type(e).__name__}: {e}")

print("\nTest 5: Testing ensure_python_int with invalid float")
try:
    result = ensure_python_int(5.5)
    print(f"  Result: {result}")
except TypeError as e:
    print(f"  Correctly raised TypeError: {e}")
except Exception as e:
    print(f"  Error: {type(e).__name__}: {e}")

print("\nTest 6: Testing Python int() behavior with infinity")
try:
    result = int(float('inf'))
    print(f"  Python int() result: {result}")
except OverflowError as e:
    print(f"  Python int() raises OverflowError: {e}")
except Exception as e:
    print(f"  Python int() raises {type(e).__name__}: {e}")