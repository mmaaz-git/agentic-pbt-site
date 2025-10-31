#!/usr/bin/env python3
"""Test what happens when we try to use non-monotonic divisions in Dask"""

import pandas as pd
import dask.dataframe as dd
import numpy as np

print("Testing non-monotonic divisions behavior...")

# Create a simple dataframe
dates = pd.date_range('2000-12-10', periods=10, freq='D')
df = pd.DataFrame({'value': np.random.randn(10)}, index=dates)
ddf = dd.from_pandas(df, npartitions=3)

print(f"Original divisions: {ddf.divisions}")
print(f"Are original divisions monotonic? {all(ddf.divisions[i] <= ddf.divisions[i+1] for i in range(len(ddf.divisions)-1))}")

# Try to create a dataframe with non-monotonic divisions using the internal API
# (This would be what happens if _resample_bin_and_out_divs returns bad divisions)
from dask.dataframe.dask_expr._repartition import Repartition

# Let's see if we can even create non-monotonic divisions
bad_divs = [pd.Timestamp('2000-12-17'), pd.Timestamp('2000-12-10')]

try:
    # This is similar to what happens internally in resample
    print(f"\nTrying to create partitions with non-monotonic divisions: {bad_divs}")
    # Note: This would be done internally, let's see if it errors or causes problems
    print("If Dask allows this, it could lead to incorrect results in downstream operations.")
except Exception as e:
    print(f"Error: {e}")