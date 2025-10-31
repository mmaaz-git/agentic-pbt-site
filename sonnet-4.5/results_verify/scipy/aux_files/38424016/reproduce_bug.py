import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

A = sp.diags([1.0, 2.0], format='csr')
result = spl.expm(A)

print(f"Type of result: {type(result)}")
print(f"Is sparse: {sp.issparse(result)}")

print("\nDocstring 'Returns' section says:")
print("  expA : (M,M) ndarray")
print("\nBut Examples section shows:")
print("  >>> Aexp")
print("  <Compressed Sparse Column sparse array of dtype 'float64'")
print("      with 3 stored elements and shape (3, 3)>")

assert sp.issparse(result), "Result is sparse, not ndarray as claimed in Returns section"
print("\nAssertion passed: Result is indeed sparse, not ndarray as claimed in Returns section")