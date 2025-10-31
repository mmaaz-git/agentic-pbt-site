import pandas as pd
import sys

try:
    print("Testing SparseArray cumsum with fill_value=0 (default)...")
    sparse = pd.arrays.SparseArray([1, 0, 2, 0, 3])
    print(f"SparseArray: {sparse}")
    print(f"Fill value: {sparse.fill_value}")
    print(f"_null_fill_value: {sparse._null_fill_value}")

    print("\nCalling cumsum()...")
    result = sparse.cumsum()
    print(f"Result: {result}")
    print("Success!")

except RecursionError as e:
    print(f"RecursionError occurred: maximum recursion depth exceeded")
    print("This confirms the bug report")
    sys.exit(1)
except Exception as e:
    print(f"Other error occurred: {e}")
    sys.exit(1)

# Also test with explicit fill_value=None (should work)
try:
    print("\n\nTesting SparseArray cumsum with fill_value=None...")
    sparse2 = pd.arrays.SparseArray([1, 0, 2, 0, 3], fill_value=None)
    print(f"SparseArray: {sparse2}")
    print(f"Fill value: {sparse2.fill_value}")
    print(f"_null_fill_value: {sparse2._null_fill_value}")

    print("\nCalling cumsum()...")
    result2 = sparse2.cumsum()
    print(f"Result: {result2}")
    print("Success!")

except RecursionError as e:
    print(f"RecursionError occurred: maximum recursion depth exceeded")
except Exception as e:
    print(f"Other error occurred: {e}")