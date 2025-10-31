from hypothesis import given, strategies as st
import scipy.special as sp
import math

@given(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False))
def test_logit_expit_inverse(x):
    result = sp.logit(sp.expit(x))
    assert math.isclose(result, x, rel_tol=1e-6), \
        f"logit(expit({x})) = {result}, expected {x}"

# Run the test
if __name__ == "__main__":
    # Try specific failing inputs mentioned in the bug report
    failing_inputs = [27.0, 30.0, 35.0, 40.0]

    for x in failing_inputs:
        try:
            test_logit_expit_inverse(x)
            print(f"x={x}: Test PASSED")
        except AssertionError as e:
            print(f"x={x}: Test FAILED - {e}")

    # Run the full hypothesis test
    print("\nRunning full hypothesis test...")
    try:
        test_logit_expit_inverse()
        print("Hypothesis test completed without failures")
    except Exception as e:
        print(f"Hypothesis test failed: {e}")