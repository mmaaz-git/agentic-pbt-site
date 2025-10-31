import pandas.arrays as pa
import numpy as np

# Test: What should cumsum do according to documentation?
# Docs say: "The resulting SparseArray will preserve the locations of NaN values,
# but the fill value will be `np.nan` regardless."

print("Testing expected behavior of cumsum on SparseArrays\n")

# Test with normal numpy array to understand expected behavior
print("1. Normal numpy array cumsum:")
arr = np.array([0, 1, 0, 2])
print(f"Array: {arr}")
print(f"Cumsum: {arr.cumsum()}")  # Should be [0, 1, 1, 3]

print("\n2. Numpy array with NaN:")
arr_nan = np.array([0, 1, np.nan, 2])
print(f"Array: {arr_nan}")
print(f"Cumsum: {np.nancumsum(arr_nan)}")  # NaN-aware cumsum

print("\n3. What the documentation says should happen:")
print("According to docs, for SparseArray cumsum:")
print("- Cumulative sum of non-NA/null values")
print("- Skips non-NA/null values during summation")
print("- Fill value will be np.nan regardless of original fill value")

# Let's manually compute what the result should be
print("\n4. Manual calculation for SparseArray([0, 1, 0, 2], fill_value=0):")
sparse_values = np.array([0, 1, 0, 2])
print(f"Original values: {sparse_values}")
print(f"Expected cumsum: {sparse_values.cumsum()}")  # [0, 1, 1, 3]
print("According to docs, the result should be a SparseArray with:")
print("- Values: [0, 1, 1, 3]")
print("- Fill value: np.nan")

# Test what to_dense() returns
print("\n5. Understanding to_dense() behavior:")
sparse = pa.SparseArray([0, 1, 0, 2], fill_value=0)
dense = sparse.to_dense()
print(f"Original sparse: {sparse}")
print(f"to_dense() result: {dense}")
print(f"Type: {type(dense)}")

# Manual fix simulation
print("\n6. Manual fix simulation:")
print("If we do: sparse.to_dense().cumsum()")
manual_cumsum = dense.cumsum()
print(f"Result: {manual_cumsum}")
print(f"Type: {type(manual_cumsum)}")
print("Then wrap in SparseArray with fill_value=np.nan:")
try:
    result = pa.SparseArray(manual_cumsum, fill_value=np.nan)
    print(f"Final SparseArray: {result}")
    print(f"Fill value: {result.fill_value}")
except Exception as e:
    print(f"Error: {e}")

# Test if the documentation claim is correct
print("\n7. Testing documentation claim about non-NA/null values:")
print("The docs say 'Cumulative sum of non-NA/null values'")
print("But this is misleading - it should cumsum ALL values")
print("The phrase likely means it operates on arrays that may contain NA/null")

# Comparison with Series cumsum
print("\n8. Comparison with pandas Series cumsum:")
import pandas as pd
series = pd.Series([0, 1, 0, 2])
print(f"Series: {series.values}")
print(f"Series cumsum: {series.cumsum().values}")

# Check if zero is considered null in sparse context
print("\n9. Is zero considered null/NA?")
print("In sparse arrays, fill_value=0 means 0 is the default/fill value")
print("But 0 is NOT null/NA - it's a valid number")
sparse_test = pa.SparseArray([0, 1, 0, 2], fill_value=0)
print(f"_null_fill_value for fill_value=0: {sparse_test._null_fill_value}")  # False
sparse_test2 = pa.SparseArray([0, 1, np.nan, 2], fill_value=np.nan)
print(f"_null_fill_value for fill_value=nan: {sparse_test2._null_fill_value}")  # True