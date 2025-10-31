from hypothesis import given, strategies as st, settings, assume
from scipy.differentiate import derivative
import numpy as np
import math

@settings(max_examples=50)
@given(
    initial_step=st.floats(min_value=0.01, max_value=10, allow_nan=False, allow_infinity=False),
    step_factor=st.floats(min_value=0.1, max_value=5, allow_nan=False, allow_infinity=False)
)
def test_step_parameters_produce_valid_results(initial_step, step_factor):
    assume(step_factor > 0.05)
    x = 1.5
    res = derivative(np.exp, x, initial_step=initial_step, step_factor=step_factor)
    if res.success:
        expected = np.exp(x)
        assert math.isclose(res.df, expected, rel_tol=1e-5)

# Run the test
if __name__ == "__main__":
    import traceback

    print("Running hypothesis test...")
    try:
        test_step_parameters_produce_valid_results()
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        traceback.print_exc()

    # Also test specifically with step_factor=1.0
    print("\nSpecific test with step_factor=1.0:")
    try:
        res = derivative(np.exp, 1.5, initial_step=0.1, step_factor=1.0)
        print(f"Success: res.df = {res.df}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")