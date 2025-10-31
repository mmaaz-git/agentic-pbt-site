import numpy as np
import scipy.io
import scipy.sparse
import tempfile
import os
from hypothesis import given, strategies as st, settings


@given(
    st.integers(1, 10),
    st.integers(1, 10),
    st.floats(allow_nan=False, allow_infinity=False, min_value=0.01, max_value=1.0)
)
@settings(max_examples=100)
def test_harwell_boeing_roundtrip(rows, cols, density):
    arr = scipy.sparse.random(rows, cols, density=density, format='csr', dtype=np.float64)

    with tempfile.NamedTemporaryFile(suffix='.hb', delete=False) as f:
        filename = f.name

    try:
        scipy.io.hb_write(filename, arr)
        loaded = scipy.io.hb_read(filename, spmatrix=False)

        if scipy.sparse.issparse(loaded):
            loaded_dense = loaded.toarray()
        else:
            loaded_dense = loaded

        arr_dense = arr.toarray()

        assert loaded_dense.shape == arr_dense.shape
        assert np.allclose(loaded_dense, arr_dense, rtol=1e-9, atol=1e-14)
    finally:
        if os.path.exists(filename):
            os.unlink(filename)


# Run the test
if __name__ == "__main__":
    test_harwell_boeing_roundtrip()