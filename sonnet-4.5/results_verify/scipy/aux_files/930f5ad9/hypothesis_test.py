import numpy as np
from hypothesis import given, strategies as st, assume, settings
from scipy.interpolate import PPoly


@settings(max_examples=500)
@given(
    st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False), min_size=2, max_size=10),
    st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False), min_size=1, max_size=10)
)
def test_ppoly_roots_are_zeros(x_values, c_values):
    x = np.array(sorted(set(x_values)))
    assume(len(x) >= 2)

    k = len(c_values)
    c = np.array(c_values).reshape(k, 1)

    try:
        pp = PPoly(c, x)
        roots = pp.roots()

        if len(roots) > 0:
            root_values = pp(roots)
            assert np.allclose(root_values, 0, atol=1e-8), \
                f"PPoly.roots() returned {roots} but pp(roots) = {root_values}, not zeros"
    except (ValueError, np.linalg.LinAlgError):
        assume(False)

# Test with the specific failing input
print("Testing with specific failing input:")
x_values = [0.0, 1.0]
c_values = [0.0]
test_ppoly_roots_are_zeros(x_values, c_values)