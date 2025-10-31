from hypothesis import given, strategies as st, settings
from pandas.io.formats.format import format_percentiles

@given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
@settings(max_examples=1000)
def test_format_percentiles_non_integer_has_decimal(percentile):
    """
    Property from docstring: "Any non-integer is always rounded to at least 1 decimal place"
    """
    formatted = format_percentiles([percentile])
    result = formatted[0]

    percent_value = percentile * 100
    is_integer_percent = abs(percent_value - round(percent_value)) < 1e-10

    if not is_integer_percent:
        assert '.' in result or ',' in result, \
            f"Non-integer percentile {percentile} formatted without decimal: {result}"

# Run the test
if __name__ == "__main__":
    test_format_percentiles_non_integer_has_decimal()