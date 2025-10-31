import numpy as np
from numpy.polynomial import Polynomial

a = Polynomial([0, 0, 0, 0, 0, 0, 0, 1])
b = Polynomial([72, 1.75])

q, r = divmod(a, b)
reconstructed = b * q + r

print("Original a:         ", a.coef)
print("Reconstructed b*q+r:", reconstructed.trim().coef)
print("Difference:         ", reconstructed.trim().coef - a.coef)
print()
print("Expected: a == b*q + r")
print("Actual difference shows numerical error of magnitude:", np.max(np.abs(reconstructed.trim().coef - a.coef)))