import numpy as np
import numpy.ma as ma

# Test ma.unique behavior with masked values
ar1 = np.array([0, 0], dtype=np.int16)
mask1 = np.array([True, False])
mar1 = ma.array(ar1, mask=mask1)

ar2 = np.array([0, 127, 0], dtype=np.int8)
mask2 = np.array([True, False, True])
mar2 = ma.array(ar2, mask=mask2)

print("Test ma.unique:")
print("Input array 1:", mar1)
unique1 = ma.unique(mar1)
print("unique(ar1):", unique1)
print("unique(ar1) mask:", ma.getmaskarray(unique1))

print("\nInput array 2:", mar2)
unique2 = ma.unique(mar2)
print("unique(ar2):", unique2)
print("unique(ar2) mask:", ma.getmaskarray(unique2))

print("\n--- Testing the algorithm used by intersect1d ---")
# This is what intersect1d does internally (when assume_unique=False)
aux = ma.concatenate((unique1, unique2))
print("concatenated unique arrays:", aux)
print("concatenated mask:", ma.getmaskarray(aux))

aux.sort()
print("after sort:", aux)
print("after sort mask:", ma.getmaskarray(aux))

result = aux[:-1][aux[1:] == aux[:-1]]
print("final result:", result)
print("final result mask:", ma.getmaskarray(result))
print("number of masked values in result:", np.sum(ma.getmaskarray(result)))