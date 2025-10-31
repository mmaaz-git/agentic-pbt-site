from hypothesis import given, strategies as st
from scipy.differentiate import derivative
import numpy as np
import pytest

@given(st.just(0.0))
def test_derivative_should_reject_zero_step_factor(step_factor):
    """Test that derivative function should reject step_factor=0 with a ValueError"""
    with pytest.raises(ValueError, match="step_factor"):
        derivative(np.sin, 1.0, step_factor=step_factor)

if __name__ == "__main__":
    # Run the test
    test_derivative_should_reject_zero_step_factor()