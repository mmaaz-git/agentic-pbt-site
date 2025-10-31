from pandas.arrays import SparseArray

# Test case with all values equal to the fill value
arr = SparseArray([0, 0, 0], fill_value=0)

print("Testing argmin on SparseArray([0, 0, 0], fill_value=0):")
try:
    result = arr.argmin()
    print(f"argmin() returned: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\nTesting argmax on SparseArray([0, 0, 0], fill_value=0):")
try:
    result = arr.argmax()
    print(f"argmax() returned: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")