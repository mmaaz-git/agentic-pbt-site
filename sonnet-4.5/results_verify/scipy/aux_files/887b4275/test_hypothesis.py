#!/usr/bin/env python3
"""Run the hypothesis test from the bug report"""

from hypothesis import given, strategies as st, settings, assume
import numpy as np
import scipy.signal
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/scipy_env')


@given(
    signal=st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=2, max_size=50),
    divisor=st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=1, max_size=30),
)
@settings(max_examples=500)
def test_deconvolve_round_trip(signal, divisor):
    assume(len(divisor) <= len(signal))
    assume(any(abs(d) > 1e-6 for d in divisor))

    signal_arr = np.array(signal)
    divisor_arr = np.array(divisor)

    quotient, remainder = scipy.signal.deconvolve(signal_arr, divisor_arr)

    reconstructed = scipy.signal.convolve(divisor_arr, quotient) + remainder

    np.testing.assert_allclose(reconstructed, signal_arr, rtol=1e-10, atol=1e-10)

if __name__ == "__main__":
    # Test with the specific failing input
    print("Testing with specific input from bug report:")
    print("signal=[0.0, 0.0], divisor=[0.0, 1.0]")
    print()

    try:
        test_deconvolve_round_trip([0.0, 0.0], [0.0, 1.0])
        print("Test passed unexpectedly!")
    except ValueError as e:
        print(f"Test failed with ValueError: {e}")
    except Exception as e:
        print(f"Test failed with {type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    print("Running full hypothesis test (limited to 10 examples for speed):")
    print()

    # Run a limited hypothesis test
    try:
        test_deconvolve_round_trip.hypothesis.max_examples = 10
        test_deconvolve_round_trip()
        print("Hypothesis tests passed!")
    except Exception as e:
        print(f"Hypothesis test failed: {e}")