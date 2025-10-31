from pandas.arrays import SparseArray
import sys

# Set a lower recursion limit to make the error happen faster
sys.setrecursionlimit(50)

# Create a boolean sparse array (which cannot have NaN fill values)
arr = SparseArray([True], fill_value=False)
print(f"Array: {arr}")
print(f"Array type: {type(arr)}")
print(f"Array dtype: {arr.dtype}")
print(f"Fill value: {arr.fill_value}")
print(f"_null_fill_value: {arr._null_fill_value}")
print()
print("Calling cumsum()...")

try:
    result = arr.cumsum()
    print(f"Result: {result}")
except RecursionError as e:
    print(f"RecursionError: {e}")