import numpy as np
import numpy.rec

# Create a non-empty record to access the pprint method
dtype = np.dtype([('x', 'i4')])
arr = np.zeros(1, dtype=dtype).view(numpy.rec.recarray)
rec = arr[0]

# Check the docstring
print("pprint docstring:")
print(rec.pprint.__doc__)

# Check if pprint is documented in help
print("\n\nHelp for pprint:")
help(rec.pprint)