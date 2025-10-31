import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from hypothesis import given, strategies as st, settings

@given(
    n=st.integers(min_value=1, max_value=20),
    density=st.floats(min_value=0.0, max_value=1.0),
)
@settings(max_examples=200, deadline=None)
def test_spbandwidth_no_crash(n, density):
    rng = np.random.RandomState(0)
    A = sp.random(n, n, density=density, format='csr', random_state=rng)

    below, above = spl.spbandwidth(A)

    assert isinstance(below, int) and isinstance(above, int)
    assert 0 <= below < n
    assert 0 <= above < n

if __name__ == "__main__":
    test_spbandwidth_no_crash()