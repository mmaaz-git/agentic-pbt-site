import numpy as np

# This demonstrates the crash when np.rec.fromrecords is given an empty list
# Empty record arrays are valid in NumPy, but fromrecords crashes instead of
# creating an empty array

try:
    rec = np.rec.fromrecords([], names='x,y')
    print("Successfully created empty record array:", rec)
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

    # Show that empty record arrays are valid
    print("\nEmpty record arrays ARE valid in NumPy:")
    empty_rec = np.recarray(shape=(0,), dtype=[('x', int), ('y', int)])
    print(f"  Created with np.recarray: {empty_rec}")
    print(f"  Shape: {empty_rec.shape}")

    # Show that the related function fromarrays handles empty input correctly
    empty_from_arrays = np.rec.fromarrays([[], []], names='a,b')
    print(f"  Created with fromarrays: {empty_from_arrays}")
    print(f"  Shape: {empty_from_arrays.shape}")