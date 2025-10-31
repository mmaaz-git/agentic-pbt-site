# Bug Report: pydantic.deprecated.json.timedelta_isoformat Incorrect Encoding

**Target**: `pydantic.deprecated.json.timedelta_isoformat`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`timedelta_isoformat` incorrectly encodes negative timedeltas with non-zero sub-day components, producing ISO 8601 strings that represent different values than the input.

## Property-Based Test

```python
import datetime
import warnings
import re
from hypothesis import given, strategies as st, settings
from pydantic.deprecated.json import timedelta_isoformat

def parse_iso_duration(iso_str):
    is_negative = iso_str.startswith('-')
    if is_negative:
        iso_str = iso_str[1:]

    match = re.match(r'P(\d+)DT(\d+)H(\d+)M(\d+)\.(\d+)S', iso_str)
    if not match:
        return None

    days, hours, minutes, seconds, microseconds = map(int, match.groups())
    total_seconds = days * 86400 + hours * 3600 + minutes * 60 + seconds + microseconds / 1000000

    if is_negative:
        total_seconds = -total_seconds

    return total_seconds

@given(st.timedeltas())
@settings(max_examples=1000)
def test_timedelta_isoformat_value_preservation(td):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = timedelta_isoformat(td)

    parsed_total_seconds = parse_iso_duration(result)
    assert parsed_total_seconds is not None, f"Failed to parse: {result}"

    actual_total_seconds = td.total_seconds()

    assert abs(parsed_total_seconds - actual_total_seconds) < 1e-6, (
        f"Value mismatch for {td}:\n"
        f"  ISO format: {result}\n"
        f"  Parsed total seconds: {parsed_total_seconds}\n"
        f"  Actual total seconds: {actual_total_seconds}\n"
        f"  Difference: {abs(parsed_total_seconds - actual_total_seconds)}"
    )
```

**Failing input**: `datetime.timedelta(days=-1, microseconds=1)`

## Reproducing the Bug

```python
import datetime
from pydantic.deprecated.json import timedelta_isoformat

td = datetime.timedelta(days=-1, microseconds=1)
result = timedelta_isoformat(td)

print(f"Input: {td}")
print(f"Total seconds: {td.total_seconds()}")
print(f"ISO output: {result}")
print(f"Expected: approximately -86399.999999 seconds")
print(f"Actual interpretation: -86400.000001 seconds")
```

Output:
```
Input: -1 day, 0:00:00.000001
Total seconds: -86399.999999
ISO output: -P1DT0H0M0.000001S
Expected: approximately -86399.999999 seconds
Actual interpretation: -86400.000001 seconds
```

## Why This Is A Bug

Python's `timedelta` represents negative durations with negative days and positive seconds/microseconds. For example, `-86399.999999` seconds is stored as `days=-1, seconds=0, microseconds=1` (i.e., -1 day + 1 microsecond).

The current implementation constructs the ISO string by:
1. Adding a minus prefix if `td.days < 0`
2. Using `abs(td.days)` for the day component
3. Using `td.seconds` and `td.microseconds` directly (which are positive)

This produces `-P1DT0H0M0.000001S`, which means "-(1 day + 1 microsecond)" = -86400.000001 seconds, when it should represent -86399.999999 seconds.

## Fix

The fix needs to convert the timedelta to total seconds first, then decompose it properly:

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
+    days, remainder = divmod(total_seconds, 86400)
+    hours, remainder = divmod(remainder, 3600)
+    minutes, remainder = divmod(remainder, 60)
+    seconds = int(remainder)
+    microseconds = int((remainder - seconds) * 1_000_000)
+
+    return f'{"-" if is_negative else ""}P{int(days)}DT{int(hours)}H{int(minutes)}M{seconds}.{microseconds:06d}S'
```