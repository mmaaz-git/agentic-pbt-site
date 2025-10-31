import numpy as np
import scipy.io
from scipy.sparse import csc_array
import tempfile

# Create an all-zero sparse matrix (5x5)
A = csc_array(np.zeros((5, 5)))

with tempfile.NamedTemporaryFile(mode='w', suffix='.mtx', delete=False) as f:
    fname = f.name
    print(f"Writing to file: {fname}")

try:
    scipy.io.mmwrite(fname, A)
    print("Successfully wrote the matrix with mmwrite!")

    # Try to read it back
    B = scipy.io.mmread(fname)
    print(f"Successfully read back. Shape: {B.shape}, nnz: {B.nnz}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()