from hypothesis import given, strategies as st, settings, assume
import pandas as pd


@given(
    data=st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1000, max_value=1000),
        min_size=5,
        max_size=50,
    ),
    n_bins=st.integers(2, 10),
)
@settings(max_examples=200)
def test_cut_preserves_length(data, n_bins):
    """
    Property: cut should preserve the length of the input array.
    Evidence: cut bins values but doesn't add or remove elements.
    """
    assume(len(set(data)) >= 2)

    result = pd.cut(data, bins=n_bins)
    assert len(result) == len(data)

# Test with the specific failing input
if __name__ == "__main__":
    print("Testing hypothesis property...")
    try:
        test_cut_preserves_length()
        print("Hypothesis test passed")
    except Exception as e:
        print(f"Hypothesis test failed: {e}")

    print("\nTesting specific failing case...")
    data = [0.0, 0.0, 0.0, 0.0, -2.225073858507e-311]
    n_bins = 2
    print(f"Data: {data}")
    print(f"n_bins: {n_bins}")

    try:
        result = pd.cut(data, bins=n_bins)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")