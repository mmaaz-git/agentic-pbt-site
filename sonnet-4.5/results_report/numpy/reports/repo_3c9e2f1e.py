import numpy as np

# The specific matrix that demonstrates the bug
A = np.array([[     0. ,      0. , 416614.5],
              [416614.5, 416614.5, 416614.5],
              [416614.5, 416614.5, 416614.5]])

# Compute determinants
det_A = np.linalg.det(A)
det_AT = np.linalg.det(A.T)

print("Matrix A:")
print(A)
print("\nMatrix A.T:")
print(A.T)
print(f"\ndet(A)   = {det_A}")
print(f"det(A.T) = {det_AT}")
print(f"\n|det(A)|   = {abs(det_A)}")
print(f"|det(A.T)| = {abs(det_AT)}")
print(f"\nSigns match? {det_A * det_AT > 0}")
print(f"Values are equal? {np.isclose(det_A, det_AT, rtol=1e-9, atol=1e-12)}")

# Additional analysis
print(f"\nDifference: {det_A - det_AT}")
print(f"Relative difference: {abs(det_A - det_AT) / max(abs(det_A), abs(det_AT))}")

# Check matrix properties
print(f"\nMatrix rank: {np.linalg.matrix_rank(A)}")
print(f"Condition number: {np.linalg.cond(A)}")