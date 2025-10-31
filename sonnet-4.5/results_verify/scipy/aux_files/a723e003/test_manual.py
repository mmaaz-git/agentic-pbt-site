import numpy as np
import scipy.sparse as sp

A = sp.csr_array([[0, 1]])
I = sp.eye(2)

result = sp.kron(A, I).tocsr()

print(f"Result:\n{result.toarray()}")
print(f"nnz: {result.nnz}")
print(f"Actual nonzeros: {np.count_nonzero(result.toarray())}")
print(f"data array: {result.data}")
print(f"Explicit zeros: {np.sum(result.data == 0)}")

result.eliminate_zeros()
print(f"After eliminate_zeros(), nnz: {result.nnz}")