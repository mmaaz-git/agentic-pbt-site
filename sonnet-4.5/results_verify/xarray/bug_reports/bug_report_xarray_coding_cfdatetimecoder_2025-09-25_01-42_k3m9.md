# Bug Report: xarray.coding.times CFDatetimeCoder AttributeError with Out-of-Bounds Datetimes

**Target**: `xarray.coding.times.CFDatetimeCoder`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When `CFDatetimeCoder.encode()` encounters out-of-bounds datetime values that cause an `OverflowError`, it attempts to fall back to `cftime` for encoding. However, if `cftime` is not installed, this fallback raises an `AttributeError: 'NoneType' object has no attribute 'num2date'` instead of providing a more informative error message.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from xarray.core.variable import Variable
from xarray.coding.times import CFDatetimeCoder

@given(
    st.lists(
        st.integers(
            min_value=np.datetime64('1700-01-01', 'ns').astype('int64'),
            max_value=np.datetime64('2200-01-01', 'ns').astype('int64')
        ),
        min_size=1,
        max_size=50
    )
)
def test_datetime_coder_round_trip_ns(values):
    data = np.array(values, dtype='datetime64[ns]')
    original_var = Variable(('time',), data)

    coder = CFDatetimeCoder(use_cftime=False, time_unit='ns')

    encoded_var = coder.encode(original_var)
    decoded_var = coder.decode(encoded_var)

    decoded_data = decoded_var.data
    if hasattr(decoded_data, 'get_duck_array'):
        decoded_data = decoded_data.get_duck_array()

    np.testing.assert_array_equal(data, decoded_data)
```

**Failing input**: `values=[703_036_036_854_775_809, -8_520_336_000_000_000_000]`

## Reproducing the Bug

```python
import numpy as np
from xarray.core.variable import Variable
from xarray.coding.times import CFDatetimeCoder

data = np.array([703_036_036_854_775_809, -8_520_336_000_000_000_000], dtype='datetime64[ns]')
original_var = Variable(('time',), data)

coder = CFDatetimeCoder(use_cftime=False, time_unit='ns')

encoded_var = coder.encode(original_var)
```

Running this code produces:
```
AttributeError: 'NoneType' object has no attribute 'num2date'
```

## Why This Is A Bug

The `CFDatetimeCoder` is designed to handle datetime encoding/decoding. When standard numpy datetime encoding fails due to overflow, the code correctly attempts to fall back to using `cftime` (lines 1121-1122 in `times.py`). However, `_unpack_time_units_and_ref_date_cftime()` at line 312 directly calls `cftime.num2date(...)` without first checking if `cftime` is available (it may be `None` if not installed).

This violates the principle of graceful degradation: the error message should clearly indicate that `cftime` is required but not installed, rather than raising a confusing `AttributeError` about `NoneType`.

The root cause is in `xarray/coding/times.py:308-318`:

```python
def _unpack_time_units_and_ref_date_cftime(units: str, calendar: str):
    time_units, ref_date = _unpack_netcdf_time_units(units)
    ref_date = cftime.num2date(  # <- BUG: cftime may be None
        0,
        units=f"microseconds since {ref_date}",
        calendar=calendar,
        only_use_cftime_datetimes=True,
    )
    return time_units, ref_date
```

## Fix

```diff
--- a/xarray/coding/times.py
+++ b/xarray/coding/times.py
@@ -307,6 +307,10 @@ def _unpack_time_unit_and_ref_date(units: str):

 def _unpack_time_units_and_ref_date_cftime(units: str, calendar: str):
+    if cftime is None:
+        raise ImportError(
+            "cftime is required to encode out-of-bounds datetime values. "
+            "Install cftime with: pip install cftime"
+        )
     # same as _unpack_netcdf_time_units but finalizes ref_date for
     # processing in encode_cf_datetime
     time_units, ref_date = _unpack_netcdf_time_units(units)
```