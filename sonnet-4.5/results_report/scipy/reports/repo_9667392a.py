import scipy.sparse as sp
import scipy.sparse.linalg as spl

# Create a zero sparse matrix (3x3 with no non-zero elements)
A = sp.csr_matrix((3, 3))
print(f"Matrix shape: {A.shape}")
print(f"Number of non-zero elements: {A.nnz}")
print(f"Matrix data: {A.toarray()}")
print()

try:
    below, above = spl.spbandwidth(A)
    print(f"Bandwidth: lower={below}, upper={above}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")