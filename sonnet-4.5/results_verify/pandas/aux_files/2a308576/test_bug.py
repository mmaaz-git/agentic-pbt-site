import pandas.arrays as pa

# Test the failing input
arr = pa.SparseArray([1, 0, 2, 0, 3], fill_value=0)
print(f"Created SparseArray: {arr}")
print(f"Fill value: {arr.fill_value}")

try:
    result = arr.cumsum()
    print(f"Cumsum result: {result}")
except RecursionError as e:
    print(f"RecursionError occurred: {str(e)[:100]}...")
except Exception as e:
    print(f"Other error: {e}")