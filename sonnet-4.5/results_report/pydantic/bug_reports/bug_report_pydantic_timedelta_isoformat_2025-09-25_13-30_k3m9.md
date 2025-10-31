# Bug Report: pydantic.deprecated.json.timedelta_isoformat Incorrect ISO 8601 Encoding

**Target**: `pydantic.deprecated.json.timedelta_isoformat`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `timedelta_isoformat()` function produces incorrect ISO 8601 duration strings for negative timedeltas, causing the round-trip (encode → parse) to fail and return a vastly different value. A negative timedelta of -1 second becomes -172799 seconds after round-tripping.

## Property-Based Test

```python
import datetime
from hypothesis import given, strategies as st, example
from pydantic.deprecated.json import timedelta_isoformat
from isodate import parse_duration

@given(st.timedeltas(min_value=datetime.timedelta(days=-365),
                     max_value=datetime.timedelta(days=365)))
@example(datetime.timedelta(seconds=-1))
@example(datetime.timedelta(seconds=-30))
@example(datetime.timedelta(hours=-1))
def test_timedelta_isoformat_roundtrip(td):
    iso_string = timedelta_isoformat(td)
    parsed_td = parse_duration(iso_string)
    assert parsed_td == td, \
        f"Round-trip failed: {td} -> {iso_string} -> {parsed_td}"
```

**Failing input**: `datetime.timedelta(seconds=-1)`

## Reproducing the Bug

```python
import datetime
from pydantic.deprecated.json import timedelta_isoformat
from isodate import parse_duration

td_original = datetime.timedelta(seconds=-1)
print(f"Original: {td_original} (total_seconds={td_original.total_seconds()})")

iso_string = timedelta_isoformat(td_original)
print(f"ISO format: {iso_string}")

td_parsed = parse_duration(iso_string)
print(f"Parsed: {td_parsed} (total_seconds={td_parsed.total_seconds()})")

print(f"Are they equal? {td_original == td_parsed}")
```

**Output:**
```
Original: -1 day, 23:59:59 (total_seconds=-1.0)
ISO format: -P1DT23H59M59.000000S
Parsed: -2 days, 0:00:01 (total_seconds=-172799.0)
Are they equal? False
```

## Why This Is A Bug

The function claims to produce "ISO 8601 encoding for Python timedelta object" but the output cannot be correctly parsed back to the original value. The root cause is that Python internally represents `timedelta(seconds=-1)` as `days=-1, seconds=86399, microseconds=0`, and the function incorrectly interprets this as "-1 day + 23 hours + 59 minutes + 59 seconds" in the ISO format.

According to ISO 8601, `-P1DT23H59M59.000000S` means "minus (1 day + 23 hours + 59 minutes + 59 seconds)" = -172799 seconds total, not -1 second.

The correct ISO format for `timedelta(seconds=-1)` should be `-PT1S`.

## Fix

The function needs to calculate the sign and components from `total_seconds()` instead of using Python's internal representation directly:

```diff
--- a/pydantic/deprecated/json.py
+++ b/pydantic/deprecated/json.py
@@ -136,6 +136,13 @@ def timedelta_isoformat(td: datetime.timedelta) -> str:
 def timedelta_isoformat(td: datetime.timedelta) -> str:
     """ISO 8601 encoding for Python timedelta object."""
     warnings.warn('`timedelta_isoformat` is deprecated.', category=PydanticDeprecatedSince20, stacklevel=2)
-    minutes, seconds = divmod(td.seconds, 60)
-    hours, minutes = divmod(minutes, 60)
-    return f'{"-" if td.days < 0 else ""}P{abs(td.days)}DT{hours:d}H{minutes:d}M{seconds:d}.{td.microseconds:06d}S'
+    total_seconds = td.total_seconds()
+    sign = '-' if total_seconds < 0 else ''
+    total_seconds = abs(total_seconds)
+
+    days = int(total_seconds // 86400)
+    remaining = total_seconds - (days * 86400)
+    hours = int(remaining // 3600)
+    remaining -= hours * 3600
+    minutes = int(remaining // 60)
+    seconds = remaining - (minutes * 60)
+    microseconds = int((seconds % 1) * 1_000_000)
+    seconds = int(seconds)
+
+    return f'{sign}P{days}DT{hours}H{minutes}M{seconds}.{microseconds:06d}S'
```