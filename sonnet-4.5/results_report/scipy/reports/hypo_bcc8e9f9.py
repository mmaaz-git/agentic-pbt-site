from hypothesis import given, strategies as st, settings
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg

@settings(max_examples=500)
@given(
    size=st.integers(min_value=1, max_value=5),
    value=st.floats(allow_nan=False, allow_infinity=False, min_value=0.1, max_value=10)
)
def test_inv_returns_sparse(size, value):
    A = sparse.diags([value] * size, offsets=0, format='csr')
    A_inv = splinalg.inv(A)
    assert sparse.issparse(A_inv), f"inv should return sparse array, got {type(A_inv)}"

# Run the test
if __name__ == "__main__":
    test_inv_returns_sparse()