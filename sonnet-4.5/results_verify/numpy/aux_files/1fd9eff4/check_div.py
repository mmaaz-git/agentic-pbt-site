import numpy as np
from numpy.polynomial import Polynomial

# Check what _div actually is
print("_div method:", Polynomial._div)

# Verify that Polynomial uses polydiv
from numpy.polynomial.polynomial import polydiv
print("\npolydiv function:", polydiv)

# Test the edge case with direct polydiv call
a_coef = np.array([0., 1.])
b_coef = np.array([1.0, 2.22507386e-311])

print("\nCalling polydiv directly with coefficients:")
try:
    quo, rem = polydiv(a_coef, b_coef)
    print("Quotient:", quo)
    print("Remainder:", rem)
    print("Contains inf:", np.any(np.isinf(quo)) or np.any(np.isinf(rem)))
except Exception as e:
    print("Error:", e)