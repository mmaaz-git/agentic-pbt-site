import pandas as pd
from pandas.io.sas.sas7bdat import _convert_datetimes

# Test with small values that should work
series_small = pd.Series([0.0, 100.0, 1000.0])
result = _convert_datetimes(series_small, "s")
print(f"Small values work: {result.tolist()}")

# Test with large value that causes overflow
series_large = pd.Series([9.223372036854776e+18])
try:
    result = _convert_datetimes(series_large, "s")
    print(f"Large value result: {result.tolist()}")
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")