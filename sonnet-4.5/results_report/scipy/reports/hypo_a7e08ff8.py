from hypothesis import given, strategies as st, settings
import numpy as np
from scipy.differentiate import derivative

@given(
    step_factor=st.floats(min_value=1.0001, max_value=1.01, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100)
def test_step_factor_close_to_one(step_factor):
    def f(x_val):
        return x_val ** 2

    x = 1.5
    res = derivative(f, x, step_factor=step_factor, maxiter=3)

test_step_factor_close_to_one()