# Bug Report: timedelta_isoformat Incorrect Encoding of Negative Timedeltas

**Target**: `pydantic.deprecated.json.timedelta_isoformat`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `timedelta_isoformat` function incorrectly encodes negative timedeltas that have positive seconds/microseconds components, producing ISO 8601 duration strings that represent different durations than the original timedelta objects.

## Property-Based Test

```python
from datetime import timedelta
from hypothesis import given, strategies as st, example
from pydantic.deprecated.json import timedelta_isoformat
import re


def parse_iso_to_seconds(iso_str):
    """Parse ISO 8601 duration to total seconds."""
    pattern = r'^(-?)P(\d+)DT(\d+)H(\d+)M(\d+)\.(\d+)S$'
    match = re.match(pattern, iso_str)
    if not match:
        raise ValueError(f"Invalid ISO: {iso_str}")

    sign_str, days, hours, minutes, seconds, microseconds = match.groups()

    total = (
        int(days) * 86400 +
        int(hours) * 3600 +
        int(minutes) * 60 +
        int(seconds) +
        int(microseconds) / 1_000_000
    )

    return -total if sign_str == '-' else total


@given(st.timedeltas(min_value=timedelta(days=-999), max_value=timedelta(days=999)))
@example(timedelta(days=-1, seconds=1))
@example(timedelta(days=-1, hours=1))
@example(timedelta(seconds=-1))
def test_timedelta_isoformat_preserves_duration(td):
    """The ISO format should represent the same duration as the original timedelta."""
    iso = timedelta_isoformat(td)
    original_seconds = td.total_seconds()
    parsed_seconds = parse_iso_to_seconds(iso)

    tolerance = 0.000001
    assert abs(original_seconds - parsed_seconds) < tolerance, (
        f"Duration mismatch for {td}:\n"
        f"  Internal: days={td.days}, seconds={td.seconds}, microseconds={td.microseconds}\n"
        f"  Original total: {original_seconds} seconds\n"
        f"  ISO format: {iso}\n"
        f"  Parsed total: {parsed_seconds} seconds\n"
        f"  Difference: {parsed_seconds - original_seconds} seconds"
    )
```

**Failing input**: `timedelta(days=-1, seconds=1)`

## Reproducing the Bug

```python
from datetime import timedelta
from pydantic.deprecated.json import timedelta_isoformat

td = timedelta(days=-1, seconds=1)

print(f"Original timedelta: {td}")
print(f"Internal representation: days={td.days}, seconds={td.seconds}, microseconds={td.microseconds}")
print(f"Total seconds: {td.total_seconds()}")

iso = timedelta_isoformat(td)
print(f"ISO format: {iso}")
print(f"Expected: -86399 seconds")
print(f"ISO represents: -(86400 + 1) = -86401 seconds")
print(f"Difference: 2 seconds")
```

## Why This Is A Bug

Python's `timedelta` stores negative durations with negative days and positive seconds/microseconds as remainders. For example, `timedelta(days=-1, seconds=1)` represents -86399 seconds total (-1 day + 1 second = -86400 + 1 = -86399).

The current implementation applies a negative sign prefix when `td.days < 0`, then uses the absolute value of days and the raw positive values of seconds and microseconds. This produces `-P1DT0H0M1.000000S`, which is interpreted as "negative (1 day + 1 second)" = -86401 seconds, not -86399 seconds.

This violates the round-trip property: the ISO duration string does not represent the same duration as the original timedelta.

## Fix

```diff
--- a/pydantic/deprecated/json.py
+++ b/pydantic/deprecated/json.py
@@ -136,6 +136,17 @@ def decimal_encoder(dec_value: Decimal) -> Union[int, float]:
 @deprecated('`timedelta_isoformat` is deprecated.', category=None)
 def timedelta_isoformat(td: datetime.timedelta) -> str:
     """ISO 8601 encoding for Python timedelta object."""
     warnings.warn('`timedelta_isoformat` is deprecated.', category=PydanticDeprecatedSince20, stacklevel=2)
-    minutes, seconds = divmod(td.seconds, 60)
-    hours, minutes = divmod(minutes, 60)
-    return f'{"-" if td.days < 0 else ""}P{abs(td.days)}DT{hours:d}H{minutes:d}M{seconds:d}.{td.microseconds:06d}S'
+
+    # Handle negative timedeltas correctly
+    # Python stores negative durations as negative days + positive seconds/microseconds
+    # We need to convert to a fully negative representation for ISO format
+    total_seconds = td.total_seconds()
+    sign = '-' if total_seconds < 0 else ''
+    total_seconds = abs(total_seconds)
+
+    days = int(total_seconds // 86400)
+    remainder = total_seconds - (days * 86400)
+    hours = int(remainder // 3600)
+    remainder -= hours * 3600
+    minutes = int(remainder // 60)
+    remainder -= minutes * 60
+    seconds = int(remainder)
+    microseconds = int((remainder - seconds) * 1_000_000)
+
+    return f'{sign}P{days}DT{hours}H{minutes}M{seconds}.{microseconds:06d}S'
```