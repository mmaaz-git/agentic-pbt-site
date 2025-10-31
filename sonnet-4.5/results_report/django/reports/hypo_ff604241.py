from hypothesis import given, strategies as st, settings
import numpy as np
from scipy.differentiate import derivative

@given(x=st.floats(min_value=0.5, max_value=2, allow_nan=False, allow_infinity=False))
@settings(max_examples=100)
def test_step_factor_exactly_one(x):
    def f(x_val):
        return x_val ** 2

    res = derivative(f, x, step_factor=1.0, maxiter=5)

    if res.success:
        assert abs(res.df - 2 * x) < 1e-6

test_step_factor_exactly_one()