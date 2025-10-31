import pandas as pd
from pandas.io.sas.sas7bdat import _convert_datetimes

# Create a simple test series with floating point values
series = pd.Series([1.0, 2.0, 3.0])

# Test with an empty string as the unit parameter (invalid)
result = _convert_datetimes(series, '')

print(f"Result dtype: {result.dtype}")
print(f"Result values: {result.values}")
print(f"Result: {result}")