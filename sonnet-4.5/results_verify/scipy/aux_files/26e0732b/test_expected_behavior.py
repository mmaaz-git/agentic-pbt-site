import scipy.sparse as sp
import numpy as np

# Test what SHOULD be the expected behavior for a zero matrix
# According to the mathematical definition, a zero matrix has all elements zero
# The bandwidth should be (0, 0) since there are no non-zero elements off the diagonal
# (and actually no non-zero elements at all)

print("Testing expected behavior for zero matrices:")
print("-" * 50)

# Create a 3x3 zero matrix
A = sp.csr_matrix((3, 3))
print(f"Matrix A (3x3 zero matrix):")
print(A.toarray())
print(f"Non-zero elements: {A.nnz}")
print()

# Create a 1x1 zero matrix
B = sp.csr_matrix((1, 1))
print(f"Matrix B (1x1 zero matrix):")
print(B.toarray())
print(f"Non-zero elements: {B.nnz}")
print()

# Create a 5x5 zero matrix
C = sp.csr_matrix((5, 5))
print(f"Matrix C (5x5 zero matrix):")
print(C.toarray())
print(f"Non-zero elements: {C.nnz}")
print()

# For comparison: identity matrix has bandwidth (0, 0)
D = sp.eye(3)
print(f"Matrix D (3x3 identity matrix):")
print(D.toarray())
print(f"Non-zero elements: {D.nnz}")

# According to the docs, identity matrix has bandwidth (0, 0)
# A zero matrix should also have bandwidth (0, 0) since it has no elements outside the main diagonal
# (it has no non-zero elements at all)

# Test the definition: "A zero denotes no sub/super diagonal entries on that side"
# For a zero matrix, there are no non-zero sub/super diagonal entries, so bandwidth should be (0, 0)