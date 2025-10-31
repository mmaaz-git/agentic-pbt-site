from dask.utils import natural_sort_key

# Test with superscript 2
print("Testing natural_sort_key('²')...")
try:
    result = natural_sort_key('²')
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\nTesting natural_sort_key('file²name')...")
try:
    result = natural_sort_key('file²name')
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\nTesting sorted(['file²', 'file1'], key=natural_sort_key)...")
try:
    result = sorted(['file²', 'file1'], key=natural_sort_key)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Test with other Unicode digits
print("\nTesting with circled digit '①'...")
try:
    result = natural_sort_key('①')
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\nTesting with cube '³'...")
try:
    result = natural_sort_key('³')
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")