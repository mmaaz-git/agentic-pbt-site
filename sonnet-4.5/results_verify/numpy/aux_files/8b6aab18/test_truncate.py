from hypothesis import given, strategies as st, settings
import numpy.polynomial as np_poly

@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100), min_size=1, max_size=20),
    st.integers(min_value=1, max_value=20)
)
@settings(max_examples=500)
def test_truncate_size(coefs, size):
    p = np_poly.Polynomial(coefs)
    p_trunc = p.truncate(size)
    assert len(p_trunc.coef) == size, f"Expected length {size}, got {len(p_trunc.coef)}"

# Run the test
test_truncate_size()