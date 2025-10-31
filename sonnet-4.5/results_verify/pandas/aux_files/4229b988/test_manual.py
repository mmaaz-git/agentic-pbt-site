from pandas.core.dtypes.common import ensure_python_int

print("Testing with float('inf'):")
try:
    ensure_python_int(float('inf'))
except OverflowError as e:
    print(f"Bug: Got OverflowError: {e}")
    print(f"Expected: TypeError according to docstring")
except TypeError as e:
    print(f"Got TypeError as expected: {e}")

print("\nTesting with float('-inf'):")
try:
    ensure_python_int(float('-inf'))
except OverflowError as e:
    print(f"Bug: Got OverflowError: {e}")
    print(f"Expected: TypeError according to docstring")
except TypeError as e:
    print(f"Got TypeError as expected: {e}")