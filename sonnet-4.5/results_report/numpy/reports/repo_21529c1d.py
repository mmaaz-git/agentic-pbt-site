import numpy as np
import scipy.io
from scipy.sparse import csc_array
import tempfile

# Create a 10x10 sparse matrix with all zeros (empty sparse matrix)
A = csc_array(np.zeros((10, 10)))

# Create a temporary file
with tempfile.NamedTemporaryFile(mode='wb', suffix='.hb', delete=False) as f:
    fname = f.name

# Try to write the empty sparse matrix to HB format
scipy.io.hb_write(fname, A)