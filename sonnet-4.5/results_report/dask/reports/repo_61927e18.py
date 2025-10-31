from dask.dataframe.utils import valid_divisions

# Test with empty list
print("Testing valid_divisions([]):")
try:
    result = valid_divisions([])
    print(f"Result: {result}")
except IndexError as e:
    print(f"IndexError: {e}")

print()

# Test with single-element list
print("Testing valid_divisions([1]):")
try:
    result = valid_divisions([1])
    print(f"Result: {result}")
except IndexError as e:
    print(f"IndexError: {e}")

print()

# Test with two-element list (should work)
print("Testing valid_divisions([1, 2]):")
try:
    result = valid_divisions([1, 2])
    print(f"Result: {result}")
except IndexError as e:
    print(f"IndexError: {e}")