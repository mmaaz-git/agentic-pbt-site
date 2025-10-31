# Bug Report: xarray.coding.times.CFDatetimeCoder Precision Loss on Round-Trip

**Target**: `xarray.coding.times.CFDatetimeCoder`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

CFDatetimeCoder violates the documented round-trip property `coder.decode(coder.encode(variable)) == variable` for high-precision datetime values, causing silent data corruption with nanosecond-level precision loss.

## Property-Based Test

```python
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings
from xarray.core.variable import Variable
from xarray.coding.times import CFDatetimeCoder


@given(
    st.lists(
        st.datetimes(
            min_value=pd.Timestamp("2000-01-01"),
            max_value=pd.Timestamp("2050-12-31"),
        ),
        min_size=1,
        max_size=20,
    )
)
@settings(max_examples=100)
def test_datetime_coder_round_trip(datetime_list):
    datetime_arr = np.array(datetime_list, dtype="datetime64[ns]")

    encoding = {"units": "days since 2000-01-01", "calendar": "proleptic_gregorian"}
    original_var = Variable(("time",), datetime_arr, encoding=encoding)

    coder = CFDatetimeCoder()

    encoded_var = coder.encode(original_var)
    decoded_var = coder.decode(encoded_var)

    np.testing.assert_array_equal(original_var.data, decoded_var.data)
    assert original_var.dims == decoded_var.dims
```

**Failing input**: `datetime_list=[datetime.datetime(2003, 1, 1, 0, 0, 0, 1)]`

## Reproducing the Bug

```python
import numpy as np
from datetime import datetime
from xarray.core.variable import Variable
from xarray.coding.times import CFDatetimeCoder

dt = datetime(2003, 1, 1, 0, 0, 0, 1)
datetime_arr = np.array([dt], dtype="datetime64[ns]")

encoding = {"units": "days since 2000-01-01", "calendar": "proleptic_gregorian"}
original_var = Variable(("time",), datetime_arr, encoding=encoding)

coder = CFDatetimeCoder()
encoded_var = coder.encode(original_var)
decoded_var = coder.decode(encoded_var)

print(f"Original: {original_var.data[0]} ({original_var.data.view('int64')[0]} ns)")
print(f"Decoded:  {decoded_var.data[0]} ({decoded_var.data.view('int64')[0]} ns)")
print(f"Lost precision: {original_var.data.view('int64')[0] - decoded_var.data.view('int64')[0]} ns")

assert np.array_equal(original_var.data, decoded_var.data), "Round-trip failed!"
```

Output:
```
Original: 2003-01-01T00:00:00.000001000 (1041379200000001000 ns)
Decoded:  2003-01-01T00:00:00.000000976 (1041379200000000976 ns)
Lost precision: 24 ns
AssertionError: Round-trip failed!
```

## Why This Is A Bug

The `VariableCoder` base class explicitly documents (xarray/coding/common.py:30-31):

> "Subclasses should implement encode() and decode(), which should satisfy the identity ``coder.decode(coder.encode(variable)) == variable``."

CFDatetimeCoder violates this contract when encoding datetime values with sub-daily precision using "days since" units. While the code emits a warning about precision loss, it still proceeds with lossy encoding, resulting in silent data corruption.

This impacts users who:
- Work with high-precision temporal data (microsecond/nanosecond resolution)
- Save and reload xarray datasets with datetime coordinates
- Expect xarray to preserve their data faithfully

## Fix

The root cause is floating-point precision loss when encoding high-resolution times with coarse units (e.g., "days" for microsecond-precision data).

The fix should ensure either:
1. Use finer-grained units automatically to preserve precision (e.g., "microseconds since 2000-01-01")
2. Raise an error instead of silently corrupting data when precision cannot be preserved
3. Use integer encoding when possible to avoid floating-point errors

A minimal fix would be to auto-select appropriate units based on the data's resolution:

```diff
diff --git a/xarray/coding/times.py b/xarray/coding/times.py
index abc123..def456 100644
--- a/xarray/coding/times.py
+++ b/xarray/coding/times.py
@@ -1360,6 +1360,15 @@ class CFDatetimeCoder(VariableCoder):
     def encode(self, variable: Variable, name: T_Name = None) -> Variable:
         if np.issubdtype(
             variable.data.dtype, np.datetime64
         ) or contains_cftime_datetimes(variable):
             dims, data, attrs, encoding = unpack_for_encoding(variable)

             units = encoding.pop("units", None)
+
+            # Auto-select finer units if needed to preserve precision
+            if units is not None and data.dtype.kind == 'M':
+                required_resolution = np.datetime_data(data.dtype)[0]
+                if required_resolution in ['ns', 'us', 'ms']:
+                    # Upgrade to finer units to avoid precision loss
+                    units = f"{required_resolution} since " + units.split(" since ")[1]
+
             calendar = encoding.pop("calendar", None)
```