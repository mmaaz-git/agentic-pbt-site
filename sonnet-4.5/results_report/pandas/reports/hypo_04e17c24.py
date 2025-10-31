from hypothesis import given, strategies as st, settings
import pandas as pd

@given(
    st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10),
        min_size=2,
        max_size=20
    ),
    st.integers(min_value=2, max_value=10)
)
@settings(max_examples=1000)
def test_rolling_variance_always_nonnegative(data, window):
    if window > len(data):
        return
    s = pd.Series(data)
    var = s.rolling(window=window).var()
    valid = var[~var.isna()]
    assert (valid >= 0).all(), f"Found negative variance: {valid[valid < 0]}"

# Run the test
if __name__ == "__main__":
    try:
        test_rolling_variance_always_nonnegative()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
        # Try the specific failing example
        print("\nTrying specific failing example:")
        data = [3222872787.0, 0.0, 2.0, 0.0]
        window = 3
        s = pd.Series(data)
        var = s.rolling(window=window).var()
        valid = var[~var.isna()]
        print(f"Data: {data}")
        print(f"Window: {window}")
        print(f"Rolling variance: {var.tolist()}")
        print(f"Valid variances: {valid.tolist()}")
        negative_vars = valid[valid < 0]
        if len(negative_vars) > 0:
            print(f"Negative variances found: {negative_vars.tolist()}")