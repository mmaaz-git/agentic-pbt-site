import numpy as np
import numpy.linalg as la
from hypothesis import given, strategies as st, settings

def matrices(min_size=1, max_size=5):
    n = st.integers(min_value=min_size, max_value=max_size)
    return n.flatmap(lambda size: st.lists(
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        min_size=size*size,
        max_size=size*size
    ).map(lambda vals: np.array(vals).reshape(size, size)))

@settings(max_examples=300)
@given(matrices(min_size=2, max_size=5))
def test_pinv_reconstruction(a):
    pinv_a = la.pinv(a)
    reconstructed = a @ pinv_a @ a
    assert np.allclose(reconstructed, a, rtol=1e-4, atol=1e-7)

if __name__ == "__main__":
    test_pinv_reconstruction()