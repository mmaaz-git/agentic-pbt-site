from hypothesis import given, settings, strategies as st
import scipy.special
import numpy as np

@given(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False))
@settings(max_examples=1000)
def test_expit_logit_round_trip(x):
    result = scipy.special.logit(scipy.special.expit(x))
    assert np.isclose(result, x, rtol=1e-9, atol=1e-9), \
        f"logit(expit({x})) = {result}, expected {x}"

if __name__ == "__main__":
    # Test with specific value x=20.0
    print("Testing with x=20.0:")
    x = 20.0
    result = scipy.special.logit(scipy.special.expit(x))
    try:
        assert np.isclose(result, x, rtol=1e-9, atol=1e-9), \
            f"logit(expit({x})) = {result}, expected {x}"
        print("Test passed for x=20.0")
    except AssertionError as e:
        print(f"Test failed: {e}")

    # Test with more values
    print("\nTesting with various values:")
    test_values = [-100, -50, -20, -10, 10, 20, 30, 40, 50, 100]
    for x in test_values:
        result = scipy.special.logit(scipy.special.expit(x))
        try:
            assert np.isclose(result, x, rtol=1e-9, atol=1e-9)
            print(f"x={x:4}: PASSED")
        except AssertionError:
            if np.isfinite(result):
                error = abs(result - x)
                print(f"x={x:4}: FAILED - result={result:.10f}, error={error:.2e}")
            else:
                print(f"x={x:4}: FAILED - result={result}")