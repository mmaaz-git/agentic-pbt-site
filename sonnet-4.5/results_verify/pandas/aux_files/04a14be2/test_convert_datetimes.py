import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas as pd
import numpy as np
from pandas.io.sas.sas7bdat import _convert_datetimes

# Test _convert_datetimes with the same large values
print("Testing _convert_datetimes function:")
print("="*50)

# Create a series with various values including the problematic ones
test_values = pd.Series([0, 86400, 253717920000.0, 2936550.0])

print("Testing with seconds:")
try:
    result_s = _convert_datetimes(test_values, "s")
    print(f"Success! Results:\n{result_s}")
except Exception as e:
    print(f"Error: {e}")

print("\nTesting with days:")
try:
    result_d = _convert_datetimes(test_values, "d")
    print(f"Success! Results:\n{result_d}")
except Exception as e:
    print(f"Error: {e}")

# Test with NaN
test_nan = pd.Series([float('nan'), 0, 100])
print("\nTesting with NaN values:")
try:
    result_nan = _convert_datetimes(test_nan, "s")
    print(f"Success! Results:\n{result_nan}")
except Exception as e:
    print(f"Error: {e}")

# Check if _convert_datetimes is actually used in the codebase
print("\n" + "="*50)
print("Checking usage of _convert_datetimes in SAS7BDATReader:")
print("="*50)

import inspect
source = inspect.getsource(pd.io.sas.sas7bdat.SAS7BDATReader)
if "_convert_datetimes" in source:
    print("_convert_datetimes IS used in SAS7BDATReader")
    # Find the line where it's used
    for i, line in enumerate(source.split('\n')):
        if '_convert_datetimes' in line:
            print(f"Line {i}: {line.strip()}")
else:
    print("_convert_datetimes is NOT used in SAS7BDATReader")

if "_parse_datetime" in source:
    print("_parse_datetime IS used in SAS7BDATReader")
else:
    print("_parse_datetime is NOT used in SAS7BDATReader")