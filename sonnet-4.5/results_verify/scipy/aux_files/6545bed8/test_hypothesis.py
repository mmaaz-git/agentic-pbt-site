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
    assert sparse.issparse(A_inv), f"inv should return sparse array for size={size}, value={value}, got {type(A_inv)}"

# Run the test
print("Running hypothesis test...")
try:
    test_inv_returns_sparse()
    print("All tests passed!")
except AssertionError as e:
    print(f"Test failed: {e}")
except Exception as e:
    print(f"Test error: {e}")