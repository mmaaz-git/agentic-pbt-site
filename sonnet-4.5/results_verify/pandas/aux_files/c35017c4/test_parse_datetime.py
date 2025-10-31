import pandas as pd
from pandas.io.sas.sas7bdat import _parse_datetime
import numpy as np

# Test with NaN
print(f"NaN input: {_parse_datetime(np.nan, 's')}")

# Test with large value
large_value = 9.223372036854776e+18
try:
    result = _parse_datetime(large_value, "s")
    print(f"Large value worked: {result}")
except Exception as e:
    print(f"Large value exception - {type(e).__name__}: {e}")