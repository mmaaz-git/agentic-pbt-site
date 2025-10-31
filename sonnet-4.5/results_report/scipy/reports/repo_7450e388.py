import scipy.sparse as sp
import scipy.sparse.linalg as spl
import numpy as np

# Test case 1: 1x1 sparse matrix
print("=" * 60)
print("Test case 1: 1x1 sparse matrix")
print("=" * 60)
A = sp.csr_matrix([[2.0]])
print(f"Input matrix A:")
print(f"  Type: {type(A)}")
print(f"  Shape: {A.shape}")
print(f"  Value: {A.toarray()}")
print(f"  Is sparse: {sp.issparse(A)}")

A_inv = spl.inv(A)
print(f"\nResult of inv(A):")
print(f"  Type: {type(A_inv)}")
print(f"  Shape: {A_inv.shape}")
print(f"  Is sparse: {sp.issparse(A_inv)}")
print(f"  Value: {A_inv}")

# Test case 2: 2x2 sparse matrix for comparison
print("\n" + "=" * 60)
print("Test case 2: 2x2 sparse matrix (for comparison)")
print("=" * 60)
B = sp.csr_matrix([[2.0, 0.0], [0.0, 3.0]])
print(f"Input matrix B:")
print(f"  Type: {type(B)}")
print(f"  Shape: {B.shape}")
print(f"  Value:\n{B.toarray()}")
print(f"  Is sparse: {sp.issparse(B)}")

B_inv = spl.inv(B)
print(f"\nResult of inv(B):")
print(f"  Type: {type(B_inv)}")
print(f"  Shape: {B_inv.shape}")
print(f"  Is sparse: {sp.issparse(B_inv)}")
print(f"  Value:\n{B_inv.toarray()}")

# Test case 3: Demonstrating the issue is in spsolve
print("\n" + "=" * 60)
print("Test case 3: Direct spsolve call with 1x1 identity")
print("=" * 60)
A = sp.csr_matrix([[2.0]])
I = sp.eye(1, format='csr')
print(f"Identity matrix I:")
print(f"  Type: {type(I)}")
print(f"  Shape: {I.shape}")
print(f"  Value: {I.toarray()}")

result = spl.spsolve(A, I)
print(f"\nResult of spsolve(A, I):")
print(f"  Type: {type(result)}")
print(f"  Shape: {result.shape}")
print(f"  Is sparse: {sp.issparse(result)}")
print(f"  Value: {result}")

# Expected behavior
print("\n" + "=" * 60)
print("EXPECTED BEHAVIOR for 1x1 case:")
print("=" * 60)
print("  Type: Should be <class 'scipy.sparse._csr.csr_matrix'>")
print("  Shape: Should be (1, 1)")
print("  Is sparse: Should be True")
print("  Value: Should be [[0.5]] as a sparse matrix")