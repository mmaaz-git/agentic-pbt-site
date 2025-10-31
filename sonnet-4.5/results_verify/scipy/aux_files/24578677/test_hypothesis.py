from hypothesis import given, settings, strategies as st, reproduce_failure
from scipy import optimize
import numpy as np
import warnings

@settings(max_examples=500)
@given(
    scale=st.floats(min_value=0.1, max_value=10, allow_nan=False, allow_infinity=False),
    offset=st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False),
)
def test_curve_fit_covariance_shape(scale, offset):
    def model(x, a, b):
        return a * x + b

    xdata = np.linspace(-10, 10, 50)
    ydata = model(xdata, scale, offset)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore expected warnings
            popt, pcov = optimize.curve_fit(model, xdata, ydata)

        assert popt.shape == (2,), f"popt shape {popt.shape}, expected (2,)"
        assert pcov.shape == (2, 2), f"pcov shape {pcov.shape}, expected (2, 2)"
        assert np.all(np.isfinite(popt)), "popt contains non-finite values"
        assert np.all(np.isfinite(pcov)), f"pcov contains non-finite values for scale={scale}, offset={offset}"
    except (ValueError, RuntimeError):
        pass  # Expected exceptions are OK

# Run the test
if __name__ == "__main__":
    # Test the specific failing case
    print("Testing failing case: scale=1.015625, offset=0.0")
    def model(x, a, b):
        return a * x + b

    xdata = np.linspace(-10, 10, 50)
    ydata = model(xdata, 1.015625, 0.0)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = optimize.curve_fit(model, xdata, ydata)
        assert np.all(np.isfinite(pcov)), f"pcov contains non-finite values"
        print("Test passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")

    # Run hypothesis tests
    print("\nRunning hypothesis tests...")
    test_curve_fit_covariance_shape()