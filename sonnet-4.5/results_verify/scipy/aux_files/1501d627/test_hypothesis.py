import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/scipy_env')

from hypothesis import given, strategies as st, settings, assume
from scipy.optimize.cython_optimize import _zeros

def polynomial_f(x, args):
    a0, a1, a2, a3 = args
    return ((a3 * x + a2) * x + a1) * x + a0

@given(
    a0=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    a1=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
    a2=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
    a3=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
    xa=st.floats(min_value=-100, max_value=0, allow_nan=False, allow_infinity=False),
    xb=st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False),
    mitr=st.integers(min_value=10, max_value=1000),
)
@settings(max_examples=500)
def test_output_validity(a0, a1, a2, a3, xa, xb, mitr):
    assume(xa < xb)

    args = (a0, a1, a2, a3)
    f_xa = polynomial_f(xa, args)
    f_xb = polynomial_f(xb, args)

    assume(f_xa * f_xb < 0)

    xtol, rtol = 1e-6, 1e-6

    output = _zeros.full_output_example(args, xa, xb, xtol, rtol, mitr)

    assert output['error_num'] >= 0, f"error_num should be non-negative, got {output['error_num']}"

# Test with the specific failing input
print("Testing with the specific failing input...")
test_output_validity.hypothesis.inner(0.0, 0.0, 0.0, 1.0, -1.0, 2.0, 10)
print("Test passed for the specific input")