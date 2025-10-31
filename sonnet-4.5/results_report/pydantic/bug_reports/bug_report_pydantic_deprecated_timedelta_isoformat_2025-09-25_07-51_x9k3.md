# Bug Report: pydantic.deprecated.json.timedelta_isoformat ISO 8601 Encoding

**Target**: `pydantic.deprecated.json.timedelta_isoformat`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `timedelta_isoformat` function produces incorrect ISO 8601 duration strings for negative timedeltas where `days < 0` but `seconds > 0` or `microseconds > 0`. The encoded duration cannot be correctly round-tripped back to the original timedelta value.

## Property-Based Test

```python
import warnings
from datetime import timedelta
from hypothesis import given, strategies as st, settings


def parse_iso_duration_to_seconds(iso_str):
    if iso_str.startswith('-P'):
        is_negative = True
        iso_str = iso_str[2:]
    elif iso_str.startswith('P'):
        is_negative = False
        iso_str = iso_str[1:]
    else:
        raise ValueError(f"Invalid ISO duration: {iso_str}")

    if 'T' in iso_str:
        date_part, time_part = iso_str.split('T')
    else:
        date_part = iso_str
        time_part = ''

    days = 0
    if 'D' in date_part:
        days = int(date_part.split('D')[0])

    hours = minutes = seconds = microseconds = 0
    if time_part:
        if 'H' in time_part:
            hours = int(time_part.split('H')[0])
            time_part = time_part.split('H')[1]
        if 'M' in time_part:
            if 'S' in time_part:
                minutes_str = time_part.split('M')[0]
                if minutes_str:
                    minutes = int(minutes_str)
                time_part = time_part.split('M')[1]
            else:
                minutes = int(time_part.split('M')[0])
                time_part = ''
        if 'S' in time_part:
            sec_str = time_part.split('S')[0]
            if '.' in sec_str:
                sec_part, microsec_part = sec_str.split('.')
                seconds = int(sec_part)
                microseconds = int(microsec_part)
            else:
                seconds = int(sec_str)

    total_seconds = days * 86400 + hours * 3600 + minutes * 60 + seconds + microseconds / 1000000
    if is_negative:
        total_seconds = -total_seconds

    return total_seconds


@settings(max_examples=1000)
@given(st.timedeltas(min_value=timedelta(days=-365), max_value=timedelta(days=365)))
def test_timedelta_isoformat_roundtrip(td):
    from pydantic.deprecated.json import timedelta_isoformat

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        iso = timedelta_isoformat(td)

    reconstructed_seconds = parse_iso_duration_to_seconds(iso)
    original_seconds = td.total_seconds()

    assert abs(reconstructed_seconds - original_seconds) < 0.000001
```

**Failing input**: `timedelta(days=-1, microseconds=1)`

## Reproducing the Bug

```python
import warnings
from datetime import timedelta
from pydantic.deprecated.json import timedelta_isoformat

td = timedelta(days=-1, microseconds=1)

print(f"timedelta: {td}")
print(f"  days={td.days}, seconds={td.seconds}, microseconds={td.microseconds}")
print(f"  total_seconds()={td.total_seconds()}")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    iso = timedelta_isoformat(td)

print(f"  ISO format: {iso}")
print(f"\nInterpretation of ISO format '{iso}':")
print(f"  Should represent: -(1 day + 0h + 0m + 0s + 0.000001s) = -86400.000001 seconds")
print(f"  But actual timedelta represents: -86399.999999 seconds")
print(f"  Difference: 2.000002e-06 seconds")
```

Output:
```
timedelta: -1 day, 0:00:00.000001
  days=-1, seconds=0, microseconds=1
  total_seconds()=-86399.999999
  ISO format: -P1DT0H0M0.000001S

Interpretation of ISO format '-P1DT0H0M0.000001S':
  Should represent: -(1 day + 0h + 0m + 0s + 0.000001s) = -86400.000001 seconds
  But actual timedelta represents: -86399.999999 seconds
  Difference: 2.000002e-06 seconds
```

## Why This Is A Bug

Python's `timedelta` normalizes negative durations such that `seconds` and `microseconds` are always non-negative (0-86399 and 0-999999 respectively), with the negative sign carried only in `days`. For example, `timedelta(seconds=-1)` is stored as `timedelta(days=-1, seconds=86399)`.

The current implementation naively applies the minus sign when `days < 0`:

```python
return f'{"-" if td.days < 0 else ""}P{abs(td.days)}DT{hours:d}H{minutes:d}M{seconds:d}.{td.microseconds:06d}S'
```

This produces incorrect ISO 8601 output because it interprets the negative sign as applying to the entire duration, but then uses the already-normalized positive `seconds` and `microseconds` values. According to ISO 8601, `-P1DT0H0M0.000001S` means -(1 day + 0.000001 seconds), not (-1 day + 0.000001 seconds).

## Fix

```diff
--- a/pydantic/deprecated/json.py
+++ b/pydantic/deprecated/json.py
@@ -13,7 +13,15 @@ from pydantic_core import PydanticUndefined
 @deprecated('`timedelta_isoformat` is deprecated.', category=None)
 def timedelta_isoformat(td: datetime.timedelta) -> str:
     """ISO 8601 encoding for Python timedelta object."""
     warnings.warn('`timedelta_isoformat` is deprecated.', category=PydanticDeprecatedSince20, stacklevel=2)
-    minutes, seconds = divmod(td.seconds, 60)
+
+    total_seconds = int(td.total_seconds())
+    sign = '-' if total_seconds < 0 else ''
+    total_seconds = abs(total_seconds)
+
+    days, remainder = divmod(total_seconds, 86400)
+    hours, remainder = divmod(remainder, 3600)
+    minutes, seconds = divmod(remainder, 60)
+    microseconds = abs(td.microseconds)
+
-    hours, minutes = divmod(minutes, 60)
-    return f'{"-" if td.days < 0 else ""}P{abs(td.days)}DT{hours:d}H{minutes:d}M{seconds:d}.{td.microseconds:06d}S'
+    return f'{sign}P{days}DT{hours}H{minutes}M{seconds}.{microseconds:06d}S'
```