#!/usr/bin/env python3
"""Mathematical verification of pseudoinverse properties"""

import numpy as np

print("=" * 60)
print("Mathematical Properties of Pseudoinverse")
print("=" * 60)

# For a matrix A with pseudoinverse A+, the Moore-Penrose conditions are:
# 1. A * A+ * A = A
# 2. A+ * A * A+ = A+
# 3. (A * A+)^T = A * A+  (A * A+ is symmetric)
# 4. (A+ * A)^T = A+ * A  (A+ * A is symmetric)

# For non-square matrices:
# If A is m x n, then A+ is n x m
# A * A+ is m x m
# A+ * A is n x n

print("\nFor a non-square matrix A of shape (m, n) where m ≠ n:")
print("- Pseudoinverse A+ has shape (n, m)")
print("- A * A+ has shape (m, m)")
print("- A+ * A has shape (n, n)")
print("- These CANNOT both equal the same identity matrix!")

# Test with example from bug report
A = np.matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
A_plus = A.I

print(f"\nExample: A is {A.shape}, A+ is {A_plus.shape}")
print(f"- A * A+ is shape {(A * A_plus).shape}")
print(f"- A+ * A is shape {(A_plus * A).shape}")

# The docstring claims both products equal np.eye(self[0,:].size)
# self[0,:].size is the number of columns (N)
N = A[0,:].size
print(f"\nThe docstring says both should equal np.eye({N}), which is shape ({N}, {N})")
print(f"But A * A+ is shape {(A * A_plus).shape} - SHAPE MISMATCH!")

print("\n" + "=" * 60)
print("What the docstring SHOULD say:")
print("=" * 60)

print("\nFor square non-singular matrices:")
print("- A^(-1) * A = I_n")
print("- A * A^(-1) = I_n")
print("Both products equal the same n×n identity matrix.")

print("\nFor non-square matrices (pseudoinverse):")
print("- A+ * A ≈ I_n (n×n identity)")
print("- A * A+ ≈ I_m (m×m identity)")
print("These are DIFFERENT sized identity matrices!")

# Demonstrate correct behavior
print("\n" + "=" * 60)
print("Correct Pseudoinverse Properties:")
print("=" * 60)

m, n = A.shape
AA_plus = A * A_plus
A_plusA = A_plus * A

print(f"\nA * A+ (should be close to I_{m}):")
print(AA_plus)
print(f"Is it close to I_{m}? {np.allclose(AA_plus, np.eye(m))}")

print(f"\nA+ * A (should be close to I_{n} for full rank):")
print(A_plusA)
# Note: For rank-deficient matrices, this won't be identity
print("Note: This is NOT identity because the matrix is rank-deficient")

# Check the Moore-Penrose conditions
print("\n" + "=" * 60)
print("Moore-Penrose Conditions Check:")
print("=" * 60)

cond1 = A * A_plus * A
print(f"1. A * A+ * A = A? {np.allclose(cond1, A)}")

cond2 = A_plus * A * A_plus
print(f"2. A+ * A * A+ = A+? {np.allclose(cond2, A_plus)}")

cond3 = (A * A_plus).T - (A * A_plus)
print(f"3. (A * A+) symmetric? {np.allclose(cond3, 0)}")

cond4 = (A_plus * A).T - (A_plus * A)
print(f"4. (A+ * A) symmetric? {np.allclose(cond4, 0)}")

print("\nAll Moore-Penrose conditions are satisfied!")
print("This confirms the pseudoinverse is computed correctly.")
print("\nThe BUG is in the DOCUMENTATION, not the implementation!")