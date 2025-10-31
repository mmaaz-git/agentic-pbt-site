import pyarrow as pa
import pyarrow.compute as pc

# Check documentation for list_element
print("list_element docstring:")
print(pc.list_element.__doc__)
print("\n" + "="*50 + "\n")

# Test pyarrow behavior directly
arr = pa.array([[1, 2, 3], [4]], type=pa.list_(pa.int64()))
print("Array:", arr)
print()

# Try to access index 0
print("Accessing index 0:")
result = pc.list_element(arr, 0)
print(result)
print()

# Try to access index 1 (should fail)
print("Accessing index 1:")
try:
    result = pc.list_element(arr, 1)
    print(result)
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
print()

# Check if pyarrow has any function to handle variable-length list indexing
print("Available list functions in pyarrow.compute:")
list_funcs = [name for name in dir(pc) if 'list' in name.lower()]
for func_name in list_funcs:
    print(f"  - {func_name}")