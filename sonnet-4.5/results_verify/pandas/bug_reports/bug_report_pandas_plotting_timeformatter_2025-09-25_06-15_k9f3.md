# Bug Report: pandas.plotting TimeFormatter Microsecond Overflow

**Target**: `pandas.plotting._matplotlib.converter.TimeFormatter.__call__`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

TimeFormatter crashes with ValueError when formatting time values where rounding produces 1000000 microseconds (out of valid range 0..999999).

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.plotting._matplotlib.converter import TimeFormatter


@given(st.floats(min_value=86400, max_value=172800, allow_nan=False, allow_infinity=False))
def test_timeformatter_wraps_at_24_hours(seconds_since_midnight):
    formatter = TimeFormatter(locs=[])
    result = formatter(seconds_since_midnight)

    s = int(seconds_since_midnight)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    _, h = divmod(h, 24)

    assert 0 <= h < 24
```

**Failing input**: `86400.99999999997`

## Reproducing the Bug

```python
from pandas.plotting._matplotlib.converter import TimeFormatter

formatter = TimeFormatter(locs=[])
x = 86400.99999999997

result = formatter(x)
```

Output:
```
ValueError: microsecond must be in 0..999999
```

## Why This Is A Bug

The TimeFormatter.__call__ method is documented to handle "seconds since 00:00 (midnight), with up to microsecond precision." When the fractional part is very close to 1.0, the rounding operation `round((x - s) * 10**6)` can produce 1000000, which exceeds the valid range for datetime.time's microsecond parameter (0..999999).

The bug occurs because:
1. `x = 86400.99999999997`
2. `s = int(x) = 86400`
3. `msus = round((x - s) * 10**6) = round(999999.99997) = 1000000`
4. `pydt.time(h, m, s, msus)` fails because `msus` must be < 1000000

This is a valid input that should be handled correctly.

## Fix

```diff
--- a/pandas/plotting/_matplotlib/converter.py
+++ b/pandas/plotting/_matplotlib/converter.py
@@ -210,6 +210,10 @@ class TimeFormatter(Formatter):
         fmt = "%H:%M:%S.%f"
         s = int(x)
         msus = round((x - s) * 10**6)
+        if msus >= 1000000:
+            s += 1
+            msus = 0
+
         ms = msus // 1000
         us = msus % 1000
         m, s = divmod(s, 60)
```

This fix handles the overflow case by incrementing the second counter and resetting microseconds to 0, which is the correct behavior when rounding pushes the microseconds to 1000000.