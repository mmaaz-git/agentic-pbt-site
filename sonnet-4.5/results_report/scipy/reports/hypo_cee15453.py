from scipy.io.arff._arffread import DateAttribute
from hypothesis import given, strategies as st, assume


@given(st.text(alphabet='MdHms-/: ', min_size=1, max_size=50))
def test_date_format_without_year_shouldnt_set_year_unit(pattern_body):
    """
    Test that date patterns without year components don't incorrectly
    set datetime_unit to 'Y' during processing.

    This test exposes a bug where 'elif "yy":' (always True) causes
    datetime_unit to be incorrectly set to 'Y' for patterns without years.
    """
    assume('y' not in pattern_body.lower())
    assume(any(x in pattern_body for x in ['M', 'd', 'H', 'm', 's']))

    pattern_str = f"date {pattern_body}"

    try:
        date_fmt, datetime_unit = DateAttribute._get_date_format(pattern_str)
        assert datetime_unit != "Y", \
            f"Pattern {pattern_body} has no year but datetime_unit is 'Y'"
    except ValueError:
        pass  # Some patterns may be invalid, that's OK


if __name__ == "__main__":
    # Run the test
    test_date_format_without_year_shouldnt_set_year_unit()