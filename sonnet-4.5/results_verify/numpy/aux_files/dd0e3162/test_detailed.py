import numpy as np
from numpy.polynomial import polynomial

# Let's trace through the computation step by step
c1 = np.array([2.0])
c2 = np.array([0.0, 5e-324])
c3 = np.array([0.5])

print("Testing denormal number behavior:")
print("5e-324 is denormal:", 5e-324 < np.finfo(float).tiny)
print("np.finfo(float).tiny =", np.finfo(float).tiny)
print("Smallest positive normal float =", np.finfo(float).tiny)
print("Smallest positive denormal float =", np.finfo(float).smallest_subnormal)
print()

# Step by step for left-to-right
print("Left-to-right: (c1 * c2) * c3")
print("Step 1: c1 * c2")
print("  [2.0] * [0.0, 5e-324] = [0.0, 1e-323]")
intermediate1 = polynomial.polymul(c1, c2)
print("  Actual result:", intermediate1)
print("  1e-323 is denormal:", 1e-323 < np.finfo(float).tiny)

print("Step 2: [0.0, 1e-323] * [0.5]")
print("  Expected: [0.0, 5e-324]")
result1 = polynomial.polymul(intermediate1, c3)
print("  Actual result:", result1)
print()

# Step by step for right-to-left
print("Right-to-left: c1 * (c2 * c3)")
print("Step 1: c2 * c3")
print("  [0.0, 5e-324] * [0.5] = ?")
intermediate2 = polynomial.polymul(c2, c3)
print("  Actual result:", intermediate2)
print("  Note: 5e-324 * 0.5 = 2.5e-324")

# Let's check what happens to 2.5e-324
test_val = 5e-324 * 0.5
print("  5e-324 * 0.5 in Python:", test_val)
print("  Is this exactly 0.0?:", test_val == 0.0)

print("Step 2: [2.0] * [0.0]")
print("  Expected: [0.0]")
result2 = polynomial.polymul(c1, intermediate2)
print("  Actual result:", result2)
print()

# Check the polytrim behavior
print("Understanding polytrim:")
print("polymul likely trims trailing zeros from polynomials")
print("When 2.5e-324 underflows to 0.0, it gets trimmed")
print("But 5e-324 is preserved as a denormal number")