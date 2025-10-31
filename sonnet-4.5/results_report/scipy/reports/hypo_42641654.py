from hypothesis import given, strategies as st, settings, assume
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

@given(st.integers(min_value=1, max_value=10), st.random_module())
@settings(max_examples=100)
def test_inv_double_inversion(n, random):
    A_dense = np.random.rand(n, n) + np.eye(n) * 2
    A = sp.csc_array(A_dense)
    assume(np.linalg.det(A_dense) > 0.01)

    Ainv = sla.inv(A)
    Ainvinv = sla.inv(Ainv)

    result = Ainvinv.toarray()
    expected = A.toarray()
    assert np.allclose(result, expected, rtol=1e-5, atol=1e-8)

if __name__ == "__main__":
    test_inv_double_inversion()