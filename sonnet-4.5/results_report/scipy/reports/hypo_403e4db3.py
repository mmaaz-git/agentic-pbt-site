from hypothesis import given, settings, strategies as st, assume
from scipy.optimize import ridder

@given(
    st.floats(min_value=-10, max_value=-0.1, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.1, max_value=10, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_ridder_converges_with_custom_tolerance(a, b):
    assume(abs(a - b) > 1e-6)

    def f(x):
        return x * x - 2.0

    fa, fb = f(a), f(b)
    assume(fa * fb < 0)

    result = ridder(f, a, b, xtol=1e-3, rtol=1e-3, full_output=True, disp=False)
    root, info = result

    assert info.converged, f"ridder failed to converge for interval [{a}, {b}]"
    assert abs(f(root)) < 1e-6, f"f(root) = {f(root)}, expected ~0"

# Run the test
test_ridder_converges_with_custom_tolerance()