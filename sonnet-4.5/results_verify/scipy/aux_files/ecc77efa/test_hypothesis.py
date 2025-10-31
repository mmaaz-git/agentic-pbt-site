import numpy as np
from hypothesis import given, strategies as st, settings, assume
from scipy.odr import Data, Model, ODR


def linear_func(beta, x):
    return beta[0] * x + beta[1]


def test_specific_case():
    """Test the specific failing case"""
    slope = 1.0
    intercept = 2.225073858507e-311
    n_points = 5

    x = np.linspace(0, 10, n_points)
    y_exact = slope * x + intercept

    data = Data(x, y_exact)
    model = Model(linear_func)

    initial_guess = [slope * 0.9, intercept * 0.9] if intercept != 0 else [slope * 0.9, 1.0]

    try:
        odr_obj = ODR(data, model, beta0=initial_guess)
        result = odr_obj.run()

        if np.any(np.isnan(result.beta)):
            print(f"Test failed: Result contains NaN: {result.beta}")
            return False
        else:
            print(f"Test passed: Result is {result.beta}")
            return True
    except Exception as e:
        print(f"Exception during test: {e}")
        return False


@given(
    slope=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
    intercept=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
    n_points=st.integers(min_value=5, max_value=50)
)
@settings(max_examples=10, deadline=5000)
def test_perfect_fit_recovery(slope, intercept, n_points):
    assume(abs(slope) > 1e-6)

    x = np.linspace(0, 10, n_points)
    y_exact = slope * x + intercept

    data = Data(x, y_exact)
    model = Model(linear_func)

    initial_guess = [slope * 0.9, intercept * 0.9] if intercept != 0 else [slope * 0.9, 1.0]

    try:
        odr_obj = ODR(data, model, beta0=initial_guess)
        result = odr_obj.run()

        assert not np.any(np.isnan(result.beta)), \
            f"Result contains NaN: {result.beta}"
    except Exception as e:
        if "Iteration limit reached" in str(e) or "not full rank" in str(e):
            assume(False)
        else:
            raise

# Run the test
if __name__ == "__main__":
    # Test the specific failing input
    print("Testing specific failing input from bug report...")
    test_specific_case()

    print("\nRunning limited Hypothesis tests...")
    try:
        test_perfect_fit_recovery()
        print("Hypothesis tests completed")
    except Exception as e:
        print(f"Hypothesis tests failed: {e}")