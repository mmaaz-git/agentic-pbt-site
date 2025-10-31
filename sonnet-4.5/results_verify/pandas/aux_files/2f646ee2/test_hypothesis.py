from hypothesis import given, strategies as st, settings
from pandas.io.sas.sas7bdat import _parse_datetime

@given(
    st.floats(
        min_value=-1e8,
        max_value=1e8,
        allow_nan=False,
        allow_infinity=False
    )
)
@settings(max_examples=200)
def test_parse_datetime_days_monotonicity(days):
    smaller = days
    larger = days + 1.0

    result_smaller = _parse_datetime(smaller, 'd')
    result_larger = _parse_datetime(larger, 'd')

    assert result_larger > result_smaller

if __name__ == "__main__":
    test_parse_datetime_days_monotonicity()
    print("Test completed without errors")