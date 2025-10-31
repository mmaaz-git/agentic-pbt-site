"""Property-based test demonstrating the is_subperiod reflexivity bug using Hypothesis"""

from hypothesis import given, strategies as st, settings
from pandas.tseries.frequencies import is_subperiod, is_superperiod

# Define test frequency strings
FREQ_STRINGS = [
    "ns", "us", "ms", "s", "min", "h",
    "D", "B", "C",
    "W", "M", "Q", "Y",
    "Y-JAN", "Y-FEB", "Q-JAN", "Q-FEB",
]

@given(st.sampled_from(FREQ_STRINGS))
@settings(max_examples=200)
def test_subperiod_reflexivity(freq):
    """Test that is_subperiod satisfies reflexivity: is_subperiod(x, x) should always be True"""
    assert is_subperiod(freq, freq), \
        f"is_subperiod({freq!r}, {freq!r}) should be True (reflexivity)"

@given(st.sampled_from(FREQ_STRINGS), st.sampled_from(FREQ_STRINGS))
@settings(max_examples=1000)
def test_subperiod_superperiod_inverse(source, target):
    """Test that is_subperiod and is_superperiod have an inverse relationship"""
    result_sub = is_subperiod(source, target)
    result_super = is_superperiod(target, source)
    assert result_sub == result_super, \
        f"is_subperiod({source!r}, {target!r}) = {result_sub}, but is_superperiod({target!r}, {source!r}) = {result_super}"

# Run the tests
if __name__ == "__main__":
    print("Testing reflexivity property of is_subperiod...")
    print("=" * 70)
    try:
        test_subperiod_reflexivity()
        print("✓ All reflexivity tests passed")
    except AssertionError as e:
        print(f"✗ Reflexivity test FAILED")
        print(f"  {e}")

    print("\nTesting inverse relationship between is_subperiod and is_superperiod...")
    print("=" * 70)
    try:
        test_subperiod_superperiod_inverse()
        print("✓ All inverse relationship tests passed")
    except AssertionError as e:
        print(f"✗ Inverse relationship test FAILED")
        print(f"  {e}")