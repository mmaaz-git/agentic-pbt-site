import scipy.sparse as sp
import scipy.sparse.linalg as spl

A = sp.csr_matrix((3, 3))
print(f"Created sparse matrix A with shape {A.shape}, nnz={A.nnz}")
print(f"Matrix:\n{A.toarray()}")

try:
    below, above = spl.spbandwidth(A)
    print(f"spbandwidth returned: below={below}, above={above}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")