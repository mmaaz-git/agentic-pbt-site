from hypothesis import given, strategies as st
from pandas.tseries.frequencies import is_subperiod, is_superperiod

# Define all valid frequency strings for testing
freq_strings = st.sampled_from([
    'D', 'B', 'C', 'h', 'min', 's', 'ms', 'us', 'ns',
    'M', 'BM', 'W', 'Y', 'Q',
    'W-MON', 'W-TUE', 'W-WED', 'W-THU', 'W-FRI', 'W-SAT', 'W-SUN',
    'Q-JAN', 'Q-FEB', 'Q-MAR', 'Q-APR', 'Q-MAY', 'Q-JUN',
    'Q-JUL', 'Q-AUG', 'Q-SEP', 'Q-OCT', 'Q-NOV', 'Q-DEC',
    'Y-JAN', 'Y-FEB', 'Y-MAR', 'Y-APR', 'Y-MAY', 'Y-JUN',
    'Y-JUL', 'Y-AUG', 'Y-SEP', 'Y-OCT', 'Y-NOV', 'Y-DEC',
])

@given(freq_strings, freq_strings)
def test_subperiod_superperiod_symmetry_strings(source, target):
    """Test that is_superperiod(a, b) == is_subperiod(b, a) for all frequency pairs."""
    result_super = is_superperiod(source, target)
    result_sub = is_subperiod(target, source)
    assert result_super == result_sub, (
        f"Symmetry violated for source='{source}', target='{target}': "
        f"is_superperiod('{source}', '{target}') = {result_super}, "
        f"is_subperiod('{target}', '{source}') = {result_sub}"
    )

if __name__ == "__main__":
    # Run the property-based test
    test_subperiod_superperiod_symmetry_strings()