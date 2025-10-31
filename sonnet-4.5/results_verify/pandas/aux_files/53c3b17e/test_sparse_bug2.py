#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

import numpy as np
from pandas.arrays import SparseArray

# Direct test without hypothesis wrapper
print("Direct test on data=[0]:")
data = [0]
arr = SparseArray(data, fill_value=0)
dense = arr.to_dense()

print(f"  SparseArray: {arr}")
print(f"  Dense array: {dense}")

try:
    sparse_min = arr.argmin()
    print(f"  SparseArray.argmin(): {sparse_min}")
except Exception as e:
    print(f"  SparseArray.argmin() ERROR: {type(e).__name__}: {e}")

try:
    dense_min = dense.argmin()
    print(f"  Dense.argmin(): {dense_min}")
except Exception as e:
    print(f"  Dense.argmin() ERROR: {type(e).__name__}: {e}")

try:
    sparse_max = arr.argmax()
    print(f"  SparseArray.argmax(): {sparse_max}")
except Exception as e:
    print(f"  SparseArray.argmax() ERROR: {type(e).__name__}: {e}")

try:
    dense_max = dense.argmax()
    print(f"  Dense.argmax(): {dense_max}")
except Exception as e:
    print(f"  Dense.argmax() ERROR: {type(e).__name__}: {e}")

# Test what happens with regular ndarray
print("\nRegular NumPy ndarray([0]):")
np_arr = np.array([0])
print(f"  argmin: {np_arr.argmin()}")
print(f"  argmax: {np_arr.argmax()}")