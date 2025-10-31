from hypothesis import given, strategies as st, settings, example
from pandas.io.sas.sas7bdat import _parse_datetime
import pandas as pd
from datetime import datetime

@given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10))
@example(2936550.0)  # The failing case from the report
@settings(max_examples=100, deadline=None)
def test_parse_datetime_handles_overflow(sas_days):
    """Test that _parse_datetime handles overflow gracefully."""
    try:
        result = _parse_datetime(sas_days, unit="d")
        # If we get a result, it should be a datetime or pd.NaT
        assert result is pd.NaT or isinstance(result, datetime), f"Expected datetime or pd.NaT, got {type(result)}"
    except OverflowError:
        # This should not happen - the function should handle overflow gracefully
        raise AssertionError(f"_parse_datetime raised OverflowError for input {sas_days}")
    except ValueError as e:
        # Only acceptable ValueError is for invalid unit
        if "unit must be 'd' or 's'" not in str(e):
            raise

if __name__ == "__main__":
    test_parse_datetime_handles_overflow()