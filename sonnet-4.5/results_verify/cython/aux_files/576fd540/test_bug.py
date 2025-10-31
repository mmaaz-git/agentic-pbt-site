#!/usr/bin/env python3
"""Test the ensure_python_int function bug report"""

import numpy as np
from pandas.core.dtypes.common import ensure_python_int

print("Testing ensure_python_int with float('inf')...")
print("-" * 50)

# Test 1: Basic reproduction test
print("\nTest 1: Basic reproduction with float('inf')")
try:
    result = ensure_python_int(float('inf'))
    print(f"Result: {result}")
except OverflowError as e:
    print(f"Got OverflowError: {e}")
except TypeError as e:
    print(f"Got TypeError: {e}")
except Exception as e:
    print(f"Got unexpected exception {type(e).__name__}: {e}")

# Test 2: Test with negative infinity
print("\nTest 2: Test with float('-inf')")
try:
    result = ensure_python_int(float('-inf'))
    print(f"Result: {result}")
except OverflowError as e:
    print(f"Got OverflowError: {e}")
except TypeError as e:
    print(f"Got TypeError: {e}")
except Exception as e:
    print(f"Got unexpected exception {type(e).__name__}: {e}")

# Test 3: Test with NaN
print("\nTest 3: Test with float('nan')")
try:
    result = ensure_python_int(float('nan'))
    print(f"Result: {result}")
except OverflowError as e:
    print(f"Got OverflowError: {e}")
except TypeError as e:
    print(f"Got TypeError: {e}")
except Exception as e:
    print(f"Got unexpected exception {type(e).__name__}: {e}")

# Test 4: Test with valid float that equals an integer
print("\nTest 4: Test with 5.0 (valid float that equals integer)")
try:
    result = ensure_python_int(5.0)
    print(f"Result: {result}, type: {type(result)}")
except Exception as e:
    print(f"Got exception {type(e).__name__}: {e}")

# Test 5: Test with float that doesn't equal an integer
print("\nTest 5: Test with 5.5 (float that doesn't equal integer)")
try:
    result = ensure_python_int(5.5)
    print(f"Result: {result}")
except OverflowError as e:
    print(f"Got OverflowError: {e}")
except TypeError as e:
    print(f"Got TypeError: {e}")
except Exception as e:
    print(f"Got unexpected exception {type(e).__name__}: {e}")

# Test 6: Test what Python's int() does with infinity
print("\nTest 6: What does Python's int() do with infinity?")
try:
    result = int(float('inf'))
    print(f"Result: {result}")
except OverflowError as e:
    print(f"Got OverflowError from int(): {e}")
except Exception as e:
    print(f"Got other exception from int() {type(e).__name__}: {e}")