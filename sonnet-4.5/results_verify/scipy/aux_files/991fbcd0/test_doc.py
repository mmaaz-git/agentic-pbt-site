import scipy.sparse as sp
import numpy as np

# Create a simple CSR matrix
data = np.array([1, 2, 3])
indices = np.array([0, 2, 1])
indptr = np.array([0, 2, 3])
A = sp.csr_matrix((data, indices, indptr), shape=(2, 3))

# Check what attributes are public
print("Checking attribute properties...")
print(f"Type of indices: {type(A.indices)}")
print(f"Is indices writable: {A.indices.flags.writeable}")

# Check docstrings
print("\n--- Documentation for CSR matrix attributes ---")
print(f"CSR matrix class docstring excerpt about data attributes:")
print(sp.csr_matrix.__doc__[:2000] if sp.csr_matrix.__doc__ else "No docstring")