from pandas.arrays import SparseArray

# Test case that should crash with all fill values
arr = SparseArray([0, 0, 0], fill_value=0)
print(f"Created SparseArray: {arr}")
print(f"Array values: {arr.to_numpy()}")
print(f"Fill value: {arr.fill_value}")

try:
    result = arr.argmin()
    print(f"argmin() returned: {result}")
except Exception as e:
    print(f"argmin() crashed with {type(e).__name__}: {e}")

try:
    result = arr.argmax()
    print(f"argmax() returned: {result}")
except Exception as e:
    print(f"argmax() crashed with {type(e).__name__}: {e}")