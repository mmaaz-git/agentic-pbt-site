from hypothesis import given, strategies as st
from pandas.io.sas.sas7bdat import _convert_datetimes
import pandas as pd

@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=100)
)
def test_convert_datetimes_seconds_no_crash(values):
    series = pd.Series(values)
    result = _convert_datetimes(series, "s")
    assert len(result) == len(values)

# Test with the specific failing input
if __name__ == "__main__":
    print("Testing with failing input...")
    values = [9.223372036854776e+18]
    series = pd.Series(values)
    try:
        result = _convert_datetimes(series, "s")
        print(f"Result: {result}")
    except Exception as e:
        print(f"Exception: {type(e).__name__}: {e}")