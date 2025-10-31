import pandas.tseries.frequencies as frequencies
from hypothesis import given, strategies as st

@given(freq1=st.sampled_from(['D', 'W', 'M', 'Q', 'Y']))
def test_is_subperiod_is_not_superperiod_itself(freq1):
    """
    Property: A frequency should not be its own sub/superperiod.

    Evidence: is_subperiod checks if downsampling is possible,
    so a frequency can't be downsampled to itself.
    """
    result = frequencies.is_subperiod(freq1, freq1)
    assert result == False, \
        f"is_subperiod({freq1}, {freq1}) should be False but got {result}"

if __name__ == "__main__":
    test_is_subperiod_is_not_superperiod_itself()