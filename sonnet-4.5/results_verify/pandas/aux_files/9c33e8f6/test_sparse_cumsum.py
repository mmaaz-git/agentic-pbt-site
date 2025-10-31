import pandas as pd
import numpy as np
import sys

print("Testing SparseArray cumsum with non-null fill value...")

# Test case from bug report
data = np.array([0, 1, 2])
sparse_arr = pd.arrays.SparseArray(data, fill_value=0)

print(f"Original array: {sparse_arr}")
print(f"Fill value: {sparse_arr.fill_value}")

try:
    result = sparse_arr.cumsum()
    print(f"Result: {result}")
except RecursionError as e:
    print(f"RecursionError occurred: {str(e)[:100]}...")
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")
    sys.exit(1)

print("Test passed - no recursion error!")