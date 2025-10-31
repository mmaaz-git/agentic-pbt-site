from pandas.core.dtypes.common import ensure_python_int

# Test NaN specifically
print("Testing ensure_python_int(float('nan')):")
try:
    result = ensure_python_int(float('nan'))
    print(f"  Result: {result}")
except Exception as e:
    print(f"  Exception type: {type(e).__name__}")
    print(f"  Exception message: {e}")
    print(f"  Exception chain:")
    if e.__cause__:
        print(f"    Caused by: {type(e.__cause__).__name__}: {e.__cause__}")