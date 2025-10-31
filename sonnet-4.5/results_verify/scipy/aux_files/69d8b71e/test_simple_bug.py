import numpy as np
import scipy.io
from scipy.sparse import csc_array
import tempfile

# Create an all-zero sparse matrix (10x10)
A = csc_array(np.zeros((10, 10)))

with tempfile.NamedTemporaryFile(mode='wb', suffix='.hb', delete=False) as f:
    fname = f.name
    print(f"Writing to file: {fname}")

try:
    scipy.io.hb_write(fname, A)
    print("Successfully wrote the matrix!")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()