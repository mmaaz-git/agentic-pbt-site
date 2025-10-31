import numpy as np
import scipy.differentiate
from hypothesis import given, strategies as st, settings
import pytest


@given(
    x=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100)
def test_derivative_zero_initial_step_error(x):
    """
    Property: initial_step=0 should raise an error during input validation.
    """
    def f(x_val):
        return x_val**2

    with pytest.raises(ValueError):
        scipy.differentiate.derivative(f, x, initial_step=0)


@given(
    x=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
    initial_step=st.floats(min_value=-10, max_value=-0.001, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100)
def test_derivative_negative_initial_step_error(x, initial_step):
    """
    Property: negative initial_step should raise an error.
    """
    def f(x_val):
        return x_val**2

    with pytest.raises(ValueError):
        scipy.differentiate.derivative(f, x, initial_step=initial_step)

# Try running the tests to see what happens
print("Testing with initial_step=0...")
try:
    test_derivative_zero_initial_step_error(0.0)
    print("Test passed - ValueError was raised as expected")
except Exception as e:
    print(f"Test failed: {e}")

print("\nTesting with negative initial_step...")
try:
    test_derivative_negative_initial_step_error(0.0, -0.5)
    print("Test passed - ValueError was raised as expected")
except Exception as e:
    print(f"Test failed: {e}")