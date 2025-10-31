import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from hypothesis import given, strategies as st, settings

@given(n=st.integers(min_value=2, max_value=10))
@settings(max_examples=50, deadline=None)
def test_expm_return_type_matches_docs(n):
    A = sp.csr_matrix((n, n))
    result = spl.expm(A)

    assert isinstance(result, np.ndarray), \
        f"Documentation says expm returns ndarray, but got {type(result)}"

if __name__ == "__main__":
    test_expm_return_type_matches_docs()