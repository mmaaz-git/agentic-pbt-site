import numpy as np
import numpy.rec

data = [9_223_372_036_854_775_808]
arr1 = np.array(data)
arr2 = np.array(data)

dtype = np.dtype([('a', 'i8'), ('b', 'i8')])
rec = numpy.rec.fromarrays([arr1, arr2], dtype=dtype)

print(f"Original: {arr1[0]}")
print(f"After fromarrays: {rec.a[0]}")
print(f"Data corrupted: {arr1[0] != rec.a[0]}")