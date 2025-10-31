import pandas.tseries.frequencies as freq
from hypothesis import given, strategies as st, settings

VALID_FREQS = ["D", "h", "B", "C", "M", "W", "Q", "Y", "min", "s", "ms", "us", "ns"]

@given(st.sampled_from(VALID_FREQS))
@settings(max_examples=50)
def test_subperiod_reflexive(freq_str):
    """
    Property: Reflexivity - is_subperiod(X, X) should always be True
    (a frequency is a subperiod of itself).
    """
    result = freq.is_subperiod(freq_str, freq_str)
    assert result, f"Reflexivity violated: is_subperiod({freq_str}, {freq_str}) returned False"

@given(st.sampled_from(VALID_FREQS))
@settings(max_examples=50)
def test_superperiod_reflexive(freq_str):
    """
    Property: Reflexivity - is_superperiod(X, X) should always be True
    (a frequency is a superperiod of itself).
    """
    result = freq.is_superperiod(freq_str, freq_str)
    assert result, f"Reflexivity violated: is_superperiod({freq_str}, {freq_str}) returned False"

if __name__ == "__main__":
    print("Testing subperiod reflexivity...")
    try:
        test_subperiod_reflexive()
        print("✓ All subperiod tests passed")
    except AssertionError as e:
        print(f"✗ Subperiod test failed: {e}")

    print("\nTesting superperiod reflexivity...")
    try:
        test_superperiod_reflexive()
        print("✓ All superperiod tests passed")
    except AssertionError as e:
        print(f"✗ Superperiod test failed: {e}")