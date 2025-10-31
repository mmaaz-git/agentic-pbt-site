from pandas.core.dtypes.common import ensure_python_int

print("Testing ensure_python_int with float('inf'):")
try:
    result = ensure_python_int(float('inf'))
    print(f"Unexpectedly succeeded with result: {result}")
except TypeError as e:
    print(f"Got TypeError (expected per docstring): {e}")
except OverflowError as e:
    print(f"Got OverflowError (actual behavior): {e}")
except Exception as e:
    print(f"Got unexpected exception {type(e).__name__}: {e}")

print("\nTesting ensure_python_int with float('-inf'):")
try:
    result = ensure_python_int(float('-inf'))
    print(f"Unexpectedly succeeded with result: {result}")
except TypeError as e:
    print(f"Got TypeError (expected per docstring): {e}")
except OverflowError as e:
    print(f"Got OverflowError (actual behavior): {e}")
except Exception as e:
    print(f"Got unexpected exception {type(e).__name__}: {e}")