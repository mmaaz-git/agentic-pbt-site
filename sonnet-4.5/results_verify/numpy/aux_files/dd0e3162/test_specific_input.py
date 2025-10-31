import numpy as np
from numpy.polynomial import polynomial

c1 = np.array([2.0])
c2 = np.array([0.0, 5e-324])
c3 = np.array([0.5])

print("Input values:")
print("c1 =", c1)
print("c2 =", c2)
print("c3 =", c3)
print()

# Test associativity: (c1 * c2) * c3 vs c1 * (c2 * c3)
c1_c2 = polynomial.polymul(c1, c2)
print("c1 * c2 =", c1_c2)

result_ltr = polynomial.polymul(c1_c2, c3)
print("(c1 * c2) * c3 =", result_ltr)

c2_c3 = polynomial.polymul(c2, c3)
print("c2 * c3 =", c2_c3)

result_rtl = polynomial.polymul(c1, c2_c3)
print("c1 * (c2 * c3) =", result_rtl)

print()
print("Are they equal?", np.array_equal(result_ltr, result_rtl))
print("Shapes: result_ltr =", result_ltr.shape, ", result_rtl =", result_rtl.shape)

# Let's also test with allclose
try:
    np.testing.assert_allclose(result_ltr, result_rtl, rtol=1e-9, atol=1e-9)
    print("Results are close within tolerance")
except AssertionError as e:
    print("Results are NOT close within tolerance:")
    print(e)