import pyarrow as pa
from pandas.core.arrays.arrow import ArrowExtensionArray

# Test the direct examples from the bug report
print("Testing ArrowExtensionArray with null array...")

# Create an array with only None
arr = ArrowExtensionArray(pa.array([None]))
print(f"Array created: {arr}")
print(f"Array dtype: {arr.dtype}")

# Test all() method
print("\nTesting all(skipna=True)...")
try:
    result_all = arr.all(skipna=True)
    print(f"Result: {result_all}")
except TypeError as e:
    print(f"TypeError raised: {e}")

# Test any() method
print("\nTesting any(skipna=True)...")
try:
    result_any = arr.any(skipna=True)
    print(f"Result: {result_any}")
except TypeError as e:
    print(f"TypeError raised: {e}")

# Test with skipna=False
print("\nTesting all(skipna=False)...")
try:
    result_all_no_skip = arr.all(skipna=False)
    print(f"Result: {result_all_no_skip}")
except TypeError as e:
    print(f"TypeError raised: {e}")

print("\nTesting any(skipna=False)...")
try:
    result_any_no_skip = arr.any(skipna=False)
    print(f"Result: {result_any_no_skip}")
except TypeError as e:
    print(f"TypeError raised: {e}")

# Test with empty array
print("\n\nTesting with empty array...")
empty_arr = ArrowExtensionArray(pa.array([]))
print(f"Empty array created: {empty_arr}")
print(f"Empty array dtype: {empty_arr.dtype}")

print("\nTesting empty array all(skipna=True)...")
try:
    result_all_empty = empty_arr.all(skipna=True)
    print(f"Result: {result_all_empty}")
except Exception as e:
    print(f"Exception raised: {e}")

print("\nTesting empty array any(skipna=True)...")
try:
    result_any_empty = empty_arr.any(skipna=True)
    print(f"Result: {result_any_empty}")
except Exception as e:
    print(f"Exception raised: {e}")