from hypothesis import given, strategies as st
from pandas.io.sas.sas7bdat import _parse_datetime

@given(st.floats(allow_nan=False, allow_infinity=False, min_value=1e13, max_value=1e16))
def test_parse_datetime_handles_large_values(x):
    try:
        result = _parse_datetime(x, 's')
    except OverflowError:
        raise AssertionError(f"_parse_datetime crashed with OverflowError for value {x}")

# Run the test
if __name__ == "__main__":
    test_parse_datetime_handles_large_values()