import numpy as np
from hypothesis import given, strategies as st, settings, example
from scipy import integrate


@given(
    spacing=st.lists(st.floats(min_value=0.1, max_value=1.9, allow_nan=False, allow_infinity=False), min_size=0, max_size=3)
)
@example(spacing=[0.5])  # Specific example that should fail
@settings(max_examples=100)
def test_newton_cotes_handles_list_input(spacing):
    spacing_sorted = sorted(set(spacing))
    rn = [0.0] + [s for s in spacing_sorted] + [float(len(spacing_sorted) + 1)]

    print(f"Testing with rn = {rn}")

    try:
        an, Bn = integrate.newton_cotes(rn, equal=0)
        assert np.all(np.isfinite(an))
        print(f"  Success: weights = {an}, B = {Bn}")
    except TypeError as e:
        print(f"  Failed with TypeError: {e}")
        raise
    except Exception as e:
        print(f"  Failed with {type(e).__name__}: {e}")
        raise

# Run the test
if __name__ == "__main__":
    print("Running hypothesis test...")
    print("=" * 60)
    try:
        test_newton_cotes_handles_list_input()
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed!")

    print("\n" + "=" * 60)
    print("Testing specific failing case from bug report:")
    rn = [0, 0.5, 2]
    print(f"rn = {rn}")
    try:
        an, Bn = integrate.newton_cotes(rn, equal=0)
        print(f"Success: weights = {an}, B = {Bn}")
    except TypeError as e:
        print(f"Failed with TypeError: {e}")