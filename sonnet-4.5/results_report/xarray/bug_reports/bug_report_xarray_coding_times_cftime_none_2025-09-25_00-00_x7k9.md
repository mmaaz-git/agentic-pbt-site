# Bug Report: xarray.coding.times encode_cf_datetime Missing cftime Availability Check

**Target**: `xarray.coding.times.encode_cf_datetime`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When `encode_cf_datetime` encounters dates that pandas cannot handle (e.g., year 10000), it attempts to fall back to using cftime without checking if cftime is installed, causing an unhelpful `AttributeError` instead of a clear error message.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import numpy as np
from hypothesis import given, strategies as st, assume
from hypothesis.extra.numpy import arrays
from xarray.coding.times import encode_cf_datetime

@given(arrays(dtype=np.dtype('datetime64[s]'), shape=st.integers(min_value=5, max_value=20)))
def test_datetime_encode_decode_roundtrip_seconds(dates):
    """Encoding then decoding datetimes should approximately preserve values."""
    assume(not np.any(np.isnat(dates)))

    try:
        dates = dates.astype('datetime64[s]')
        encoded, units, calendar = encode_cf_datetime(dates, units="seconds since 2000-01-01")
    except AttributeError as e:
        if "'NoneType' object has no attribute 'num2date'" in str(e):
            raise AssertionError("Bug: cftime is None but code tries to use it")
        raise
```

**Failing input**: `array(['10000-01-01T00:00:00', '10000-01-01T00:00:00', '10000-01-01T00:00:00', '10000-01-01T00:00:00', '10000-01-01T00:00:00'], dtype='datetime64[s]')`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import numpy as np
from xarray.coding.times import encode_cf_datetime

dates = np.array(['10000-01-01T00:00:00'], dtype='datetime64[s]')

encoded, units, calendar = encode_cf_datetime(dates, units="seconds since 2000-01-01")
```

**Output:**
```
AttributeError: 'NoneType' object has no attribute 'num2date'
```

## Why This Is A Bug

When cftime is not installed, the module-level import sets `cftime = None`:

```python
try:
    import cftime
except ImportError:
    cftime = None
```

The `encode_cf_datetime` function calls `_eagerly_encode_cf_datetime`, which has a try/except block (lines 1121-1143 in times.py). When pandas-based encoding fails with `OutOfBoundsDatetime`, `OverflowError`, or `ValueError`, it falls back to cftime-based encoding at line 1122:

```python
except (OutOfBoundsDatetime, OverflowError, ValueError):
    time_units, ref_date = _unpack_time_units_and_ref_date_cftime(units, calendar)
```

This function at line 312 calls `cftime.num2date()` without checking if cftime is available, leading to an unhelpful AttributeError instead of a clear message about the missing dependency.

Expected behavior: The code should check if cftime is available before attempting to use it and provide a clear error message if it's not installed when needed.

## Fix

```diff
--- a/xarray/coding/times.py
+++ b/xarray/coding/times.py
@@ -1119,6 +1119,12 @@ def _eagerly_encode_cf_datetime(
         num = reshape(num.values, dates.shape)

     except (OutOfBoundsDatetime, OverflowError, ValueError):
+        if cftime is None:
+            raise ValueError(
+                f"Unable to encode dates to CF format with units {units!r}. "
+                "Dates are outside the range supported by pandas. "
+                "Install the 'cftime' package to handle these dates."
+            )
         time_units, ref_date = _unpack_time_units_and_ref_date_cftime(units, calendar)
         time_delta_cftime = _time_units_to_timedelta(time_units)
         needed_units = _infer_needed_units_cftime(ref_date, data_units, calendar)
```

Alternatively, the same check could be added in `_unpack_time_units_and_ref_date_cftime` itself, or both locations for defense in depth.