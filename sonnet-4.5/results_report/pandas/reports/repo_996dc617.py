from pandas.core.dtypes.common import ensure_python_int

# Test with positive infinity
print("Testing ensure_python_int with float('inf'):")
try:
    result = ensure_python_int(float('inf'))
    print(f"Result: {result}")
except OverflowError as e:
    print(f"Got OverflowError: {e}")
    print(f"Expected: TypeError according to docstring")
except TypeError as e:
    print(f"Got expected TypeError: {e}")

print()

# Test with negative infinity
print("Testing ensure_python_int with float('-inf'):")
try:
    result = ensure_python_int(float('-inf'))
    print(f"Result: {result}")
except OverflowError as e:
    print(f"Got OverflowError: {e}")
    print(f"Expected: TypeError according to docstring")
except TypeError as e:
    print(f"Got expected TypeError: {e}")