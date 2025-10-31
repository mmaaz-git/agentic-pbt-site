from pandas.core.dtypes.common import ensure_python_int

# Test with positive infinity
try:
    result = ensure_python_int(float('inf'))
    print(f"Result for inf: {result}")
except Exception as e:
    print(f"Exception for inf: {type(e).__name__}: {e}")

# Test with negative infinity
try:
    result = ensure_python_int(float('-inf'))
    print(f"Result for -inf: {result}")
except Exception as e:
    print(f"Exception for -inf: {type(e).__name__}: {e}")

# Test with NaN
try:
    result = ensure_python_int(float('nan'))
    print(f"Result for nan: {result}")
except Exception as e:
    print(f"Exception for nan: {type(e).__name__}: {e}")