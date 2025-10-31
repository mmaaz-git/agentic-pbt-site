from pandas.core.dtypes.common import ensure_python_int

# Test with positive infinity
try:
    result = ensure_python_int(float('inf'))
    print(f"float('inf'): Converted successfully to {result}")
except TypeError as e:
    print(f"float('inf'): Correctly raised TypeError: {e}")
except OverflowError as e:
    print(f"float('inf'): BUG - Raised OverflowError instead of TypeError: {e}")

# Test with negative infinity
try:
    result = ensure_python_int(float('-inf'))
    print(f"float('-inf'): Converted successfully to {result}")
except TypeError as e:
    print(f"float('-inf'): Correctly raised TypeError: {e}")
except OverflowError as e:
    print(f"float('-inf'): BUG - Raised OverflowError instead of TypeError: {e}")

# Test with NaN for comparison
try:
    result = ensure_python_int(float('nan'))
    print(f"float('nan'): Converted successfully to {result}")
except TypeError as e:
    print(f"float('nan'): Correctly raised TypeError: {e}")
except OverflowError as e:
    print(f"float('nan'): BUG - Raised OverflowError instead of TypeError: {e}")