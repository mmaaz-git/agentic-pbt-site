# Bug Report: pydantic.deprecated.json.timedelta_isoformat - Incorrect handling of negative timedeltas

**Target**: `pydantic.deprecated.json.timedelta_isoformat`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `timedelta_isoformat` function incorrectly encodes negative timedeltas to ISO 8601 format. Due to Python's internal representation of negative timedeltas (where negative durations are stored as negative days plus positive seconds), the function produces incorrect ISO strings that don't represent the actual duration.

## Property-Based Test

```python
import datetime
from hypothesis import given, strategies as st, settings
from pydantic.deprecated.json import timedelta_isoformat


@given(st.timedeltas(min_value=datetime.timedelta(days=-1000), max_value=datetime.timedelta(days=1000)))
@settings(max_examples=1000)
def test_timedelta_isoformat_total_seconds_match(td):
    result = timedelta_isoformat(td)
    total_seconds = td.total_seconds()

    import re
    match = re.match(r'^(-?)P(\d+)DT(\d+)H(\d+)M(\d+)\.(\d+)S$', result)
    sign, days, hours, minutes, seconds, microseconds = match.groups()
    sign_multiplier = -1 if sign == '-' else 1

    reconstructed_seconds = sign_multiplier * (
        int(days) * 86400 +
        int(hours) * 3600 +
        int(minutes) * 60 +
        int(seconds) +
        int(microseconds) / 1_000_000
    )

    assert abs(total_seconds - reconstructed_seconds) < 0.000001
```

**Failing input**: `datetime.timedelta(days=-1, microseconds=1)` (which is -86399.999999 seconds)

## Reproducing the Bug

```python
import datetime
from pydantic.deprecated.json import timedelta_isoformat

td = datetime.timedelta(hours=-2)
result = timedelta_isoformat(td)

assert td.total_seconds() == -7200.0
assert result == '-P1DT22H0M0.000000S'

td2 = datetime.timedelta(days=-1, microseconds=1)
assert td2.total_seconds() == -86399.999999
assert timedelta_isoformat(td2) == '-P1DT0H0M0.000001S'
```

## Why This Is A Bug

1. **For `-2 hours`**: The function outputs `-P1DT22H0M0.000000S` (negative 1 day + 22 hours), but this represents -7200 seconds incorrectly. The correct output should be `-PT2H0M0.000000S`.

2. **For `-1 day + 1 microsecond`** (which is -86399.999999 seconds): The function outputs `-P1DT0H0M0.000001S` which, when parsed, represents -86400.000001 seconds - a difference of 2 microseconds.

The bug occurs because Python's `timedelta` stores negative durations as negative days plus positive seconds/microseconds. For example, `-2 hours` is stored as `days=-1, seconds=79200` (which is -1 day + 22 hours). The function naively uses these internal attributes without converting them to the correct total duration.

This is a high-severity bug because:
- It produces **incorrect** data that violates the ISO 8601 standard semantics
- It causes **silent data corruption** when round-tripping through ISO format
- It affects all negative timedeltas with sub-day components

## Fix

Convert the timedelta to total seconds first, then decompose into ISO components:

```diff
 def timedelta_isoformat(td: datetime.timedelta) -> str:
     """ISO 8601 encoding for Python timedelta object."""
     warnings.warn('`timedelta_isoformat` is deprecated.', category=PydanticDeprecatedSince20, stacklevel=2)
-    minutes, seconds = divmod(td.seconds, 60)
-    hours, minutes = divmod(minutes, 60)
-    return f'{"-" if td.days < 0 else ""}P{abs(td.days)}DT{hours:d}H{minutes:d}M{seconds:d}.{td.microseconds:06d}S'
+    total_seconds = td.total_seconds()
+    is_negative = total_seconds < 0
+    abs_seconds = abs(total_seconds)
+
+    days = int(abs_seconds // 86400)
+    remainder = abs_seconds - (days * 86400)
+    hours = int(remainder // 3600)
+    remainder -= hours * 3600
+    minutes = int(remainder // 60)
+    remainder -= minutes * 60
+    seconds = int(remainder)
+    microseconds = int((remainder - seconds) * 1_000_000)
+
+    return f'{"-" if is_negative else ""}P{days}DT{hours}H{minutes}M{seconds}.{microseconds:06d}S'
```