from hypothesis import given, strategies as st, settings
from pandas.tseries.frequencies import is_subperiod, is_superperiod

freq_strings = st.sampled_from(['D', 'W', 'M', 'Q', 'Y', 'h', 'B'])

@given(freq_strings)
@settings(max_examples=50)
def test_is_subperiod_self_consistency(freq):
    result_sub = is_subperiod(freq, freq)
    result_super = is_superperiod(freq, freq)

    assert result_sub == result_super, \
        f"Self-inconsistent: is_subperiod({freq}, {freq})={result_sub} != is_superperiod({freq}, {freq})={result_super}"

if __name__ == "__main__":
    test_is_subperiod_self_consistency()