import scipy.sparse as sp
import scipy.sparse.linalg as spl

# Test with 1x1 matrix
A = sp.csr_matrix([[2.0]])
A_inv = spl.inv(A)

print("Test with 1x1 matrix:")
print(f"Type: {type(A_inv)}")
print(f"Shape: {A_inv.shape}")
print(f"Is sparse: {sp.issparse(A_inv)}")
print(f"Value: {A_inv}")
print()

# Test with 2x2 matrix for comparison
B = sp.csr_matrix([[2.0, 0], [0, 3.0]])
B_inv = spl.inv(B)

print("Test with 2x2 matrix:")
print(f"Type: {type(B_inv)}")
print(f"Shape: {B_inv.shape}")
print(f"Is sparse: {sp.issparse(B_inv)}")
print(f"Values:\n{B_inv.toarray()}")