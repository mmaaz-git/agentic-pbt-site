import pandas as pd
import numpy as np

# Test what the expected behavior should be
data = [1, 0, 2, 0, 3]

# Test with regular numpy array
print("Regular numpy array cumsum:")
arr = np.array(data)
print(f"Input: {arr}")
print(f"Cumsum: {arr.cumsum()}")

# Test with pandas Series
print("\nPandas Series cumsum:")
series = pd.Series(data)
print(f"Input: {list(series)}")
print(f"Cumsum: {list(series.cumsum())}")

# Test what a dense array does
print("\nDense array cumsum (what SparseArray.to_dense() gives):")
sparse = pd.arrays.SparseArray(data)
dense = sparse.to_dense()
print(f"Dense array type: {type(dense)}")
print(f"Dense array: {dense}")
# Can we call cumsum on the dense array?
try:
    dense_cumsum = dense.cumsum()
    print(f"Dense cumsum: {dense_cumsum}")
except AttributeError as e:
    print(f"Dense array doesn't have cumsum: {e}")
    print("Trying numpy cumsum:")
    dense_cumsum = np.cumsum(dense)
    print(f"numpy.cumsum(dense): {dense_cumsum}")

# Test with SparseArray with None fill value (should work)
print("\n\nSparseArray with fill_value=None (nan):")
sparse_nan = pd.arrays.SparseArray(data, fill_value=np.nan)
print(f"Fill value: {sparse_nan.fill_value}")
print(f"_null_fill_value: {sparse_nan._null_fill_value}")
try:
    result = sparse_nan.cumsum()
    print(f"Cumsum result: {result}")
    print(f"Cumsum as dense: {result.to_dense()}")
except RecursionError:
    print("RecursionError occurred!")
except Exception as e:
    print(f"Error: {e}")