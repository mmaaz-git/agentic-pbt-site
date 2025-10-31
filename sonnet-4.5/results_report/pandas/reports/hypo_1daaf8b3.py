from hypothesis import given, strategies as st, assume
from pandas.io.formats.format import format_percentiles


@given(
    percentile=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
)
def test_format_percentiles_non_integer_has_decimal(percentile):
    assume(percentile not in [0.0, 1.0])
    assume(not (percentile * 100).is_integer())

    result = format_percentiles([percentile])

    assert len(result) == 1
    value_str = result[0].rstrip('%')
    assert '.' in value_str, f"Non-integer {percentile} should have decimal: {result[0]}"


if __name__ == "__main__":
    test_format_percentiles_non_integer_has_decimal()