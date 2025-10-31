from hypothesis import given, strategies as st, settings
import numpy as np
import scipy.io
from scipy.sparse import csc_array
import tempfile
import os


@given(st.integers(min_value=1, max_value=30),
       st.integers(min_value=1, max_value=30),
       st.floats(min_value=0.1, max_value=0.9))
@settings(max_examples=50)
def test_hb_write_read_roundtrip_sparse(rows, cols, sparsity):
    np.random.seed(42)
    A_dense = np.random.random((rows, cols))
    A_dense[A_dense < sparsity] = 0
    A = csc_array(A_dense)

    with tempfile.NamedTemporaryFile(mode='wb', suffix='.hb', delete=False) as f:
        fname = f.name

    try:
        scipy.io.hb_write(fname, A)
        B = scipy.io.hb_read(fname)
        B_dense = B.toarray() if hasattr(B, 'toarray') else B
        A_dense_check = A.toarray() if hasattr(A, 'toarray') else A
        assert np.allclose(A_dense_check, B_dense, rtol=1e-10)
    finally:
        if os.path.exists(fname):
            os.remove(fname)

# Run the test
if __name__ == "__main__":
    test_hb_write_read_roundtrip_sparse()
    print("Test completed")