import pandas as pd
import numpy as np

print("Testing SparseArray cumsum with null fill value (NaN)...")

# Test with null fill value (works)
data = np.array([0.0, 1.0, 2.0, np.nan, 3.0])
sparse_arr = pd.arrays.SparseArray(data, fill_value=np.nan)

print(f"Original array: {sparse_arr}")
print(f"Fill value: {sparse_arr.fill_value}")
print(f"Is null fill value: {sparse_arr._null_fill_value}")

result = sparse_arr.cumsum()
print(f"Result: {result}")
print(f"Result values: {list(result)}")

print("\n" + "="*50 + "\n")

# Compare with dense array cumsum
dense_arr = sparse_arr.to_dense()
print(f"Dense array: {dense_arr}")
dense_cumsum = np.nancumsum(dense_arr)
print(f"Dense cumsum (nancumsum): {dense_cumsum}")

# Test what the expected result should be for non-null fill value
print("\n" + "="*50 + "\n")
print("Testing expected behavior for non-null fill value...")
data2 = np.array([0, 1, 2])
sparse_arr2 = pd.arrays.SparseArray(data2, fill_value=0)
print(f"Original array: {sparse_arr2}")
dense2 = sparse_arr2.to_dense()
print(f"Dense version: {dense2}")
expected = dense2.cumsum()
print(f"Expected cumsum result: {expected}")
print(f"Expected values: {list(expected)}")