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
        assert binned.notna().sum() == len(s)
    except ValueError:
        pass

# Test with the specific failing input
def test_specific_case():
    values = [0.0, 2.2250738585e-313]
    bins = 2
    s = pd.Series(values)

    print(f"Testing values: {values}")
    print(f"Bins: {bins}")

    try:
        binned = pd.cut(s, bins=bins)
        print(f"Result: {binned.tolist()}")
        print(f"Non-NA count: {binned.notna().sum()}")
        print(f"Expected count: {len(s)}")
        assert binned.notna().sum() == len(s), f"Expected {len(s)} non-NA values, got {binned.notna().sum()}"
    except AssertionError as e:
        print(f"Assertion failed: {e}")
        return False
    except Exception as e:
        print(f"Exception raised: {type(e).__name__}: {e}")
        return False
    return True

if __name__ == "__main__":
    # Run the specific test case
    result = test_specific_case()
    print(f"\nSpecific test case {'passed' if result else 'failed'}")

    # Run the hypothesis test
    print("\nRunning hypothesis tests...")
    try:
        test_cut_preserves_count()
        print("All hypothesis tests passed")
    except AssertionError as e:
        print(f"Hypothesis test failed: {e}")