from hypothesis import given, assume, strategies as st, settings
from scipy.optimize.cython_optimize._zeros import full_output_example
import math


def eval_polynomial(coeffs, x):
    a0, a1, a2, a3 = coeffs
    return a0 + a1*x + a2*x**2 + a3*x**3


@given(
    a0=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    a1=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    a2=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    a3=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    xa=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    xb=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=500)
def test_sign_error_when_no_sign_change(a0, a1, a2, a3, xa, xb):
    assume(xa < xb)
    assume(abs(xb - xa) > 1e-10)

    args = (a0, a1, a2, a3)
    f_xa = eval_polynomial(args, xa)
    f_xb = eval_polynomial(args, xb)

    assume(not math.isnan(f_xa) and not math.isinf(f_xa))
    assume(not math.isnan(f_xb) and not math.isinf(f_xb))
    assume(abs(f_xa) > 1e-100 and abs(f_xb) > 1e-100)

    if f_xa * f_xb > 0:
        xtol, rtol, mitr = 1e-6, 1e-6, 100
        result = full_output_example(args, xa, xb, xtol, rtol, mitr)

        assert result['error_num'] == -1, (
            f"Expected sign error (error_num=-1) when f(xa)*f(xb) > 0, "
            f"but got error_num={result['error_num']}. "
            f"f({xa})={f_xa}, f({xb})={f_xb}, Args: {args}"
        )


if __name__ == "__main__":
    test_sign_error_when_no_sign_change()