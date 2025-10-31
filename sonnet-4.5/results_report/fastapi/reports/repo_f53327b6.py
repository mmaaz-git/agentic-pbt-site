from dask.array.slicing import check_index
import numpy as np

# Test case: Boolean array too long for the dimension
bool_array = np.array([True, True, True])
dimension = 1

print(f"Boolean array size: {bool_array.size}")
print(f"Dimension size: {dimension}")
print(f"Array is {'too long' if bool_array.size > dimension else 'too short' if bool_array.size < dimension else 'correct size'}")
print()

try:
    check_index(0, bool_array, dimension)
    print("No error raised")
except IndexError as e:
    print(f"IndexError raised: {e}")
    print()
    print("Analysis:")
    if "not long enough" in str(e) and bool_array.size > dimension:
        print(f"ERROR: Message says 'not long enough' but array size {bool_array.size} > dimension size {dimension}")
        print("The array is actually TOO LONG, not too short!")