import numpy as np
import numpy.rec

# Create a record with no fields (empty dtype)
dtype = np.dtype([])
arr = np.zeros(1, dtype=dtype).view(numpy.rec.recarray)
rec = arr[0]

# Try to pretty-print the empty record
print("Attempting to call pprint() on an empty record...")
try:
    result = rec.pprint()
    print(f"Success! Result: '{result}'")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()