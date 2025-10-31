# Bug Report: pydantic.deprecated.json.timedelta_isoformat Incorrect ISO 8601 Format for Negative Timedeltas

**Target**: `pydantic.deprecated.json.timedelta_isoformat`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `timedelta_isoformat` function generates incorrect ISO 8601 duration strings for negative timedeltas with non-zero seconds or microseconds. The generated format does not round-trip correctly because the function only applies the negative sign to the days component while seconds and microseconds remain positive, violating the expected interpretation of ISO 8601 durations.

## Property-Based Test

```python
import datetime
import re
from hypothesis import given, strategies as st
from pydantic.deprecated.json import timedelta_isoformat


def parse_iso_duration(iso_str):
    pattern = r'^(-?)P(\d+)DT(\d+)H(\d+)M(\d+)\.(\d+)S$'
    match = re.match(pattern, iso_str)

    if not match:
        raise ValueError(f"Invalid ISO format: {iso_str}")

    sign, days, hours, minutes, seconds, microseconds = match.groups()
    sign_multiplier = -1 if sign == '-' else 1

    total_seconds = (
        sign_multiplier * int(days) * 86400 +
        sign_multiplier * int(hours) * 3600 +
        sign_multiplier * int(minutes) * 60 +
        sign_multiplier * int(seconds) +
        sign_multiplier * int(microseconds) / 1000000
    )

    return total_seconds


@given(st.timedeltas(
    min_value=datetime.timedelta(days=-999999),
    max_value=datetime.timedelta(days=999999)
))
def test_timedelta_isoformat_roundtrip(td):
    iso = timedelta_isoformat(td)
    original_total = td.total_seconds()
    parsed_total = parse_iso_duration(iso)

    assert abs(original_total - parsed_total) < 0.001, \
        f"Round-trip failed for {td}: {original_total} != {parsed_total}, ISO: {iso}"
```

**Failing input**: `datetime.timedelta(days=-1, seconds=1)`

## Reproducing the Bug

```python
import datetime
from pydantic.deprecated.json import timedelta_isoformat

td = datetime.timedelta(days=-1, seconds=1)
iso = timedelta_isoformat(td)

print(f"Input timedelta: {td}")
print(f"  total_seconds: {td.total_seconds()}")
print(f"ISO 8601 output: {iso}")
print(f"Expected if parsed: -{1*86400 + 1} = -86401 seconds")
print(f"Actual value: {td.total_seconds()} = -86399 seconds")
```

Output:
```
Input timedelta: -1 day, 0:00:01
  total_seconds: -86399.0
ISO 8601 output: -P1DT0H0M1.000000S
Expected if parsed: -86401 = -86401 seconds
Actual value: -86399.0 = -86399 seconds
```

## Why This Is A Bug

Python's `timedelta` normalizes negative durations such that `seconds` and `microseconds` are always in the range [0, 86400) and [0, 1000000) respectively, with `days` carrying the negative sign. For example, `-1 second` is stored as `days=-1, seconds=86399, microseconds=0` (representing -1 day + 86399 seconds = -1 second).

The current implementation only applies the negative sign to the days component, producing `-P1DT0H0M1.000000S` for the above example. According to ISO 8601, this should be interpreted as "-(1 day + 0 hours + 0 minutes + 1 second)" = -86401 seconds, but the actual timedelta represents -86399 seconds.

This breaks the fundamental property that an ISO 8601 duration string should accurately represent the underlying time duration.

## Fix

The function should convert negative timedeltas to their absolute total duration first, then format with a single negative sign:

```diff
 def timedelta_isoformat(td: datetime.timedelta) -> str:
     """ISO 8601 encoding for Python timedelta object."""
     warnings.warn('`timedelta_isoformat` is deprecated.', category=PydanticDeprecatedSince20, stacklevel=2)
-    minutes, seconds = divmod(td.seconds, 60)
-    hours, minutes = divmod(minutes, 60)
-    return f'{"-" if td.days < 0 else ""}P{abs(td.days)}DT{hours:d}H{minutes:d}M{seconds:d}.{td.microseconds:06d}S'
+    total_seconds = td.total_seconds()
+    is_negative = total_seconds < 0
+    if is_negative:
+        total_seconds = -total_seconds
+
+    days = int(total_seconds // 86400)
+    remaining_seconds = total_seconds - (days * 86400)
+    hours = int(remaining_seconds // 3600)
+    remaining_seconds -= hours * 3600
+    minutes = int(remaining_seconds // 60)
+    remaining_seconds -= minutes * 60
+    seconds = int(remaining_seconds)
+    microseconds = int((remaining_seconds - seconds) * 1000000)
+
+    return f'{"-" if is_negative else ""}P{days}DT{hours}H{minutes}M{seconds}.{microseconds:06d}S'
```