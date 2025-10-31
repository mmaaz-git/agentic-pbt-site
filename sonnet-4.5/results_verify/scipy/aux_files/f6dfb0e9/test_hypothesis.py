from hypothesis import given, strategies as st, assume, settings
import numpy as np
from scipy.differentiate import derivative

@given(
    step_factor=st.floats(min_value=0.99, max_value=1.01, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=20)
def test_step_factor_near_one(step_factor):
    """step_factor very close to 1 should either be rejected or handled gracefully."""
    assume(0.99 < step_factor < 1.01)

    def f(x):
        return np.exp(x)

    print(f"Testing step_factor={step_factor}")
    try:
        result = derivative(f, 1.0, step_factor=step_factor, order=4, maxiter=2)

        # If it doesn't raise an error, it should produce a valid result
        assert result.success or not np.isnan(result.df), \
            f"step_factor={step_factor} produced NaN without raising error"
        print(f"  Success: df={result.df}")
    except np.linalg.LinAlgError as e:
        # This is acceptable - singular matrix detected
        print(f"  LinAlgError: {e}")
    except ValueError as e:
        # This is the ideal behavior - validate step_factor != 1
        assert "step_factor" in str(e).lower()
        print(f"  ValueError (ideal): {e}")

# Run the test
test_step_factor_near_one()