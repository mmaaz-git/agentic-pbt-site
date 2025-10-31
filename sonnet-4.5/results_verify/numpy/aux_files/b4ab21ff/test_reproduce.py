import numpy as np
import numpy.rec

dtype = np.dtype([])
arr = np.zeros(1, dtype=dtype).view(numpy.rec.recarray)
rec = arr[0]

print("About to call rec.pprint()...")
try:
    result = rec.pprint()
    print(f"Success! Result: {repr(result)}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")