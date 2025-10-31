from hypothesis import given, strategies as st
from scipy.differentiate import derivative
import numpy as np
import pytest

@given(st.floats(min_value=-10, max_value=0, allow_nan=False, allow_infinity=False))
def test_derivative_should_reject_non_positive_initial_step(initial_step):
    """Test that derivative raises ValueError for non-positive initial_step."""
    print(f"Testing with initial_step={initial_step}")
    try:
        result = derivative(np.sin, 1.0, initial_step=initial_step)
        print(f"No exception raised. Result: success={result.success}, status={result.status}, df={result.df}")
        assert False, "Expected ValueError but no exception was raised"
    except ValueError as e:
        if "initial_step" in str(e):
            print(f"Got expected ValueError with 'initial_step' in message: {e}")
        else:
            print(f"Got ValueError but without 'initial_step' in message: {e}")
            raise
    except Exception as e:
        print(f"Got unexpected exception: {type(e).__name__}: {e}")
        raise

# Test with specific values
print("Testing with initial_step=0.0")
try:
    result = derivative(np.sin, 1.0, initial_step=0.0)
    print(f"No exception raised. Result: success={result.success}, status={result.status}, df={result.df}")
except ValueError as e:
    print(f"Got ValueError: {e}")

print("\nTesting with initial_step=-1.0")
try:
    result = derivative(np.sin, 1.0, initial_step=-1.0)
    print(f"No exception raised. Result: success={result.success}, status={result.status}, df={result.df}")
except ValueError as e:
    print(f"Got ValueError: {e}")