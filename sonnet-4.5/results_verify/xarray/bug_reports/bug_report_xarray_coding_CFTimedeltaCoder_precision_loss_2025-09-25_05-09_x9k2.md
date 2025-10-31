# Bug Report: xarray.coding.times.CFTimedeltaCoder Precision Loss on Round-Trip

**Target**: `xarray.coding.times.CFTimedeltaCoder`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

CFTimedeltaCoder violates the documented round-trip property `coder.decode(coder.encode(variable)) == variable` for high-precision timedelta values, causing silent data corruption with nanosecond-level precision loss.

## Property-Based Test

```python
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings
from xarray.core.variable import Variable
from xarray.coding.times import CFTimedeltaCoder


@given(
    st.lists(
        st.timedeltas(
            min_value=pd.Timedelta(seconds=0),
            max_value=pd.Timedelta(days=365),
        ),
        min_size=1,
        max_size=20,
    )
)
@settings(max_examples=100)
def test_timedelta_coder_round_trip(timedelta_list):
    timedelta_arr = np.array(timedelta_list, dtype="timedelta64[ns]")

    encoding = {"units": "seconds"}
    original_var = Variable(("time",), timedelta_arr, encoding=encoding)

    coder = CFTimedeltaCoder()

    encoded_var = coder.encode(original_var)
    decoded_var = coder.decode(encoded_var)

    np.testing.assert_array_equal(original_var.data, decoded_var.data)
    assert original_var.dims == decoded_var.dims
```

**Failing input**: `timedelta_list=[datetime.timedelta(seconds=1, microseconds=1)]`

## Reproducing the Bug

```python
import numpy as np
from datetime import timedelta
from xarray.core.variable import Variable
from xarray.coding.times import CFTimedeltaCoder

td = timedelta(seconds=1, microseconds=1)
timedelta_arr = np.array([td], dtype="timedelta64[ns]")

encoding = {"units": "seconds"}
original_var = Variable(("time",), timedelta_arr, encoding=encoding)

coder = CFTimedeltaCoder()
encoded_var = coder.encode(original_var)
decoded_var = coder.decode(encoded_var)

print(f"Original: {original_var.data[0]} ({original_var.data.view('int64')[0]} ns)")
print(f"Decoded:  {decoded_var.data[0]} ({decoded_var.data.view('int64')[0]} ns)")
print(f"Lost precision: {original_var.data.view('int64')[0] - decoded_var.data.view('int64')[0]} ns")

assert np.array_equal(original_var.data, decoded_var.data), "Round-trip failed!"
```

Output:
```
Original: 1000001000 nanoseconds (1000001000 ns)
Decoded:  1000000999 nanoseconds (1000000999 ns)
Lost precision: 1 ns
AssertionError: Round-trip failed!
```

## Why This Is A Bug

The `VariableCoder` base class explicitly documents (xarray/coding/common.py:30-31):

> "Subclasses should implement encode() and decode(), which should satisfy the identity ``coder.decode(coder.encode(variable)) == variable``."

CFTimedeltaCoder violates this contract when encoding timedelta values with nanosecond precision using "seconds" units. While the code emits a warning about precision loss, it still proceeds with lossy encoding, resulting in silent data corruption.

This impacts users who:
- Work with high-precision temporal duration data (nanosecond resolution)
- Save and reload xarray datasets with timedelta variables
- Expect xarray to preserve their data faithfully

The bug is particularly concerning because:
1. Data corruption is silent (only a warning, not an error)
2. The loss may be small (1 nanosecond) but accumulates over many values
3. Users may not notice the warning and assume their data is safe

## Fix

The root cause is floating-point precision loss when encoding high-resolution timedeltas with coarse units (e.g., "seconds" for nanosecond-precision data).

The fix should ensure either:
1. Use finer-grained units automatically to preserve precision (e.g., "nanoseconds" for ns-precision data)
2. Raise an error instead of silently corrupting data when precision cannot be preserved
3. Use integer encoding to avoid floating-point errors

A minimal fix would be to auto-select appropriate units based on the data's resolution:

```diff
diff --git a/xarray/coding/times.py b/xarray/coding/times.py
index abc123..def456 100644
--- a/xarray/coding/times.py
+++ b/xarray/coding/times.py
@@ -1480,6 +1480,14 @@ class CFTimedeltaCoder(VariableCoder):
     def encode(self, variable: Variable, name: T_Name = None) -> Variable:
         if np.issubdtype(variable.data.dtype, np.timedelta64):
             dims, data, attrs, encoding = unpack_for_encoding(variable)

             units = encoding.pop("units", None)
+
+            # Auto-select finer units if needed to preserve precision
+            if units is not None and data.dtype.kind == 'm':
+                required_resolution = np.datetime_data(data.dtype)[0]
+                if required_resolution in ['ns', 'us', 'ms'] and units in ['seconds', 'minutes', 'hours', 'days']:
+                    # Use nanoseconds to preserve precision
+                    units = required_resolution
+
             dtype = encoding.get("dtype", None)
```