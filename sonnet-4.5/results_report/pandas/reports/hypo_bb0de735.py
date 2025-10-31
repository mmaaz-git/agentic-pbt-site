from hypothesis import given, strategies as st, settings
from pandas.io.sas.sas7bdat import _parse_datetime
from datetime import datetime
import pandas as pd


@given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10))
@settings(max_examples=1000)
def test_parse_datetime_days_unit(sas_datetime):
    result = _parse_datetime(sas_datetime, unit='d')

    if not pd.isna(result):
        assert isinstance(result, datetime)


if __name__ == "__main__":
    test_parse_datetime_days_unit()