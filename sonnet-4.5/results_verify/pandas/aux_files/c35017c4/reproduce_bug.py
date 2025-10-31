import pandas as pd
from pandas.io.sas.sas7bdat import _convert_datetimes

# Test with small values
series_small = pd.Series([0.0, 100.0, 1000.0])
result = _convert_datetimes(series_small, "s")
print(f"Small values work: {result.tolist()}")

# Test with large value
series_large = pd.Series([9.223372036854776e+18])
try:
    result = _convert_datetimes(series_large, "s")
    print(f"Large value worked: {result.tolist()}")
except OverflowError as e:
    print(f"OverflowError: {e}")
except Exception as e:
    print(f"Other exception - {type(e).__name__}: {e}")