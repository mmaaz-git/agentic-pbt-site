import pandas.tseries.frequencies as freq
from hypothesis import given, strategies as st, settings


freq_strings = st.sampled_from([
    'D', 'h', 'min', 's', 'ms', 'us', 'ns',
    'W', 'B', 'ME', 'QE', 'YE', 'BME', 'BQE', 'BYE',
])


@given(source=freq_strings, target=freq_strings)
@settings(max_examples=500)
def test_is_superperiod_subperiod_inverse(source, target):
    if freq.is_superperiod(source, target):
        assert freq.is_subperiod(target, source), \
            f"is_superperiod({source!r}, {target!r}) is True but is_subperiod({target!r}, {source!r}) is False"

# Run the test
if __name__ == "__main__":
    test_is_superperiod_subperiod_inverse()