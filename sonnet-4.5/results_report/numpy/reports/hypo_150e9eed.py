import numpy as np
import numpy.lib.scimath as scimath
from hypothesis import given, strategies as st, settings

@given(
    st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    st.integers(min_value=-5, max_value=5)
)
@settings(max_examples=1000, deadline=None)
def test_power_definition(x, n):
    """Test that scimath.power never returns NaN for valid inputs."""
    result = scimath.power(x, n)

    # Check if result contains NaN
    if np.isscalar(result):
        has_nan = np.isnan(result)
    else:
        has_nan = np.any(np.isnan(result))

    # The assertion that fails
    assert not has_nan, f"scimath.power({x}, {n}) returned {result} which contains NaN"

if __name__ == "__main__":
    # Run the test
    print("Running Hypothesis test for scimath.power...")
    print("Testing that scimath.power never returns NaN for valid (non-NaN, non-inf) inputs...")
    print()

    try:
        test_power_definition()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed with assertion error:")
        print(f"  {e}")
        print()
        print("This confirms the bug: scimath.power returns NaN for certain valid inputs.")
        print()

        # Show the specific failing case
        x = -9.499558537778752e-188
        n = -2
        result = scimath.power(x, n)
        print(f"Minimal failing example:")
        print(f"  x = {x}")
        print(f"  n = {n}")
        print(f"  scimath.power(x, n) = {result}")
        print(f"  Contains NaN: {np.isnan(result)}")
        print()
        print("Expected: A valid complex or real number (likely inf or inf+0j)")
        print("Actual: inf+nanj (complex with NaN in imaginary part)")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()