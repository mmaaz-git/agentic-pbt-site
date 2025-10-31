from hypothesis import given, strategies as st, settings
from pandas.tseries import frequencies

freq_strings = st.sampled_from([
    'D', 'h', 'min', 's', 'ms', 'us', 'ns',
    'W', 'ME', 'MS', 'QE', 'QS', 'YE', 'YS',
    'B', 'BME', 'BMS', 'BQE', 'BQS', 'BYE', 'BYS'
])

@given(freq_strings, freq_strings)
@settings(max_examples=500)
def test_subperiod_superperiod_inverse_relationship(source, target):
    is_sub = frequencies.is_subperiod(source, target)
    is_super = frequencies.is_superperiod(target, source)
    assert is_sub == is_super, f"Failed for source={source}, target={target}: is_subperiod({source}, {target})={is_sub}, is_superperiod({target}, {source})={is_super}"

# Run the test
test_subperiod_superperiod_inverse_relationship()