from pandas.arrays import SparseArray
import sys
import traceback

print("Testing simple example...")
try:
    arr = SparseArray([1, 2, 3], fill_value=0)
    print(f"Created SparseArray: {arr}")
    print(f"_null_fill_value: {arr._null_fill_value}")

    result = arr.cumsum()
    print(f"Result: {result}")
except RecursionError as e:
    print("RecursionError occurred as expected:")
    print(f"Error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
    traceback.print_exc()