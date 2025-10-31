from hypothesis import given, strategies as st, settings
import pandas as pd
from pandas.io.sas.sas7bdat import _convert_datetimes

@given(st.text().filter(lambda x: x not in ['d', 's']))
@settings(max_examples=20)
def test_convert_datetimes_invalid_unit_raises(invalid_unit):
    series = pd.Series([1.0, 2.0, 3.0])

    try:
        _convert_datetimes(series, invalid_unit)
        assert False, f"Should have raised error for invalid unit '{invalid_unit}'"
    except (ValueError, KeyError) as e:
        pass

if __name__ == "__main__":
    test_convert_datetimes_invalid_unit_raises()