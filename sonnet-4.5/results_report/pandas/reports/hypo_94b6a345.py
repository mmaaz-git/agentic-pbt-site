from hypothesis import assume, given, settings, strategies as st
import pandas as pd


@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=0, max_value=1e6), min_size=2, max_size=50),
    st.integers(min_value=2, max_value=10),
)
@settings(max_examples=500)
def test_cut_preserves_count(values, bins):
    assume(len(set(values)) >= 2)
    s = pd.Series(values)

    try:
        binned = pd.cut(s, bins=bins)
        assert binned.notna().sum() == len(s), f"Expected all {len(s)} values to be binned, but got {binned.notna().sum()} non-NaN values"
    except ValueError:
        pass  # Some cases might raise ValueError, which is acceptable


if __name__ == "__main__":
    # Run the test with the specific failing input
    print("Running Hypothesis test with failing input...")
    print("values=[0.0, 2.2250738585e-313], bins=2")
    print()

    values = [0.0, 2.2250738585e-313]
    bins = 2

    # Manually test the logic from the property test
    s = pd.Series(values)
    try:
        binned = pd.cut(s, bins=bins)
        assert binned.notna().sum() == len(s), f"Expected all {len(s)} values to be binned, but got {binned.notna().sum()} non-NaN values"
        print("Test PASSED (unexpectedly)")
    except AssertionError as e:
        print(f"Test FAILED with assertion error: {e}")
    except ValueError as e:
        print(f"Test raised ValueError (acceptable): {e}")
    except Exception as e:
        print(f"Test FAILED with unexpected error: {type(e).__name__}: {e}")