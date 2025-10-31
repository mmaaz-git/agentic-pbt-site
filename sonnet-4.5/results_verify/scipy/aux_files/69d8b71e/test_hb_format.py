import scipy.io
from scipy.sparse import csc_array
import numpy as np
import tempfile

# Test reading/writing a matrix with very few non-zeros
print("Testing matrix with 1 non-zero element:")
A = csc_array(np.zeros((3, 3)))
A[1, 1] = 1.0
print(f"Matrix shape: {A.shape}, nnz: {A.nnz}")

with tempfile.NamedTemporaryFile(mode='wb', suffix='.hb', delete=False) as f:
    fname = f.name

try:
    scipy.io.hb_write(fname, A)
    print("Successfully wrote sparse matrix with 1 non-zero")
    B = scipy.io.hb_read(fname)
    print(f"Successfully read back. Shape: {B.shape}, nnz: {B.nnz}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*50)

# Test with truly empty matrix
print("Testing matrix with 0 non-zero elements:")
A_empty = csc_array(np.zeros((3, 3)))
print(f"Matrix shape: {A_empty.shape}, nnz: {A_empty.nnz}")

with tempfile.NamedTemporaryFile(mode='wb', suffix='.hb', delete=False) as f:
    fname2 = f.name

try:
    scipy.io.hb_write(fname2, A_empty)
    print("Successfully wrote empty sparse matrix")
    B_empty = scipy.io.hb_read(fname2)
    print(f"Successfully read back. Shape: {B_empty.shape}, nnz: {B_empty.nnz}")
except Exception as e:
    print(f"Error: {e}")