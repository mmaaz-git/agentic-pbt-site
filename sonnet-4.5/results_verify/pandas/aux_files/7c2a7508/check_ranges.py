#!/usr/bin/env python3

import pandas as pd
import numpy as np

# Check pandas datetime limits
print("Checking pandas datetime limits:")
print(f"pd.Timestamp.min: {pd.Timestamp.min}")
print(f"pd.Timestamp.max: {pd.Timestamp.max}")

# Convert to seconds since Unix epoch
unix_epoch = pd.Timestamp('1970-01-01')
min_seconds = (pd.Timestamp.min - unix_epoch).total_seconds()
max_seconds = (pd.Timestamp.max - unix_epoch).total_seconds()

print(f"\nSeconds from Unix epoch:")
print(f"Min seconds: {min_seconds}")
print(f"Max seconds: {max_seconds}")

# Check SAS origin
from pandas.io.sas.sas7bdat import _sas_origin, _unix_origin
td = (_sas_origin - _unix_origin).as_unit("s")
print(f"\nSAS origin: {_sas_origin}")
print(f"Unix origin: {_unix_origin}")
print(f"Time difference: {td}")

# Check the conversion with a boundary value
print("\n" + "="*50)
print("Testing boundary values:")

# Test values near the pandas limits
test_values = [
    max_seconds,  # Should work
    max_seconds + 1,  # Should fail
    9.223372036854775e15,  # Around where it starts failing
    9.223372036854776e15,  # This should fail
]

for val in test_values:
    try:
        series = pd.Series([val])
        from pandas.io.sas.sas7bdat import _convert_datetimes
        result = _convert_datetimes(series, 's')
        print(f"Value {val}: Success - {result.iloc[0]}")
    except Exception as e:
        print(f"Value {val}: {type(e).__name__}: {e}")

# Test what numpy int64 can hold
print("\n" + "="*50)
print("Numpy int64 limits:")
print(f"np.iinfo(np.int64).min: {np.iinfo(np.int64).min}")
print(f"np.iinfo(np.int64).max: {np.iinfo(np.int64).max}")

# Test cast_from_unit_vectorized directly
print("\n" + "="*50)
print("Testing cast_from_unit_vectorized directly:")
from pandas._libs.tslibs.conversion import cast_from_unit_vectorized

test_vals = [1e18, 1e19, 1e20]
for val in test_vals:
    try:
        arr = np.array([val])
        result = cast_from_unit_vectorized(arr, unit="s", out_unit="ms")
        print(f"Value {val}: Success - result = {result}")
    except Exception as e:
        print(f"Value {val}: {type(e).__name__}: {e}")