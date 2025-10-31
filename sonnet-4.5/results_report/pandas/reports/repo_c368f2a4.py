from pandas.core.dtypes.common import ensure_python_int

# Test with positive infinity
print("Testing with float('inf'):")
try:
    result = ensure_python_int(float('inf'))
    print(f"Result: {result}")
except Exception as e:
    print(f"{type(e).__name__}: {e}")

print("\nTesting with float('-inf'):")
# Test with negative infinity
try:
    result = ensure_python_int(float('-inf'))
    print(f"Result: {result}")
except Exception as e:
    print(f"{type(e).__name__}: {e}")