import numpy as np
import numpy.ma as ma

ar1 = np.array([0, 0], dtype=np.int16)
mask1 = np.array([True, False])
mar1 = ma.array(ar1, mask=mask1)

ar2 = np.array([0, 127, 0], dtype=np.int8)
mask2 = np.array([True, False, True])
mar2 = ma.array(ar2, mask=mask2)

print("Input array 1:", mar1)
print("Input array 1 mask:", mask1)
print("Input array 2:", mar2)
print("Input array 2 mask:", mask2)
print()

intersection = ma.intersect1d(mar1, mar2)
print('Result:', intersection)
print('Result data:', intersection.data)
print('Result mask:', ma.getmaskarray(intersection))
print('Number of masked values:', np.sum(ma.getmaskarray(intersection)))
print()
print("Expected: At most 1 masked value (per documentation: 'Masked values are considered equal one to the other')")
print("Actual:", np.sum(ma.getmaskarray(intersection)), "masked values")