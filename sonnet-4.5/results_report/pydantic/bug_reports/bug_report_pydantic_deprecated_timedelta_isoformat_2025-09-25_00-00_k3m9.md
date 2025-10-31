# Bug Report: timedelta_isoformat Incorrect ISO 8601 Format for Negative Durations

**Target**: `pydantic.deprecated.json.timedelta_isoformat`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `timedelta_isoformat` function produces semantically incorrect ISO 8601 duration strings for negative timedeltas. It outputs Python's internal representation rather than the actual duration, leading to formats that represent drastically different durations than intended.

## Property-Based Test

```python
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import datetime
from hypothesis import given, strategies as st
from pydantic.deprecated.json import timedelta_isoformat


def parse_iso_duration_to_seconds(iso_string):
    is_negative = iso_string.startswith('-')
    if is_negative:
        iso_string = iso_string[1:]

    if not iso_string.startswith('P'):
        raise ValueError("Invalid ISO duration")

    iso_string = iso_string[1:]
    days = hours = minutes = 0
    seconds = 0.0

    if 'D' in iso_string:
        day_part, iso_string = iso_string.split('D', 1)
        days = int(day_part)

    if 'T' in iso_string:
        iso_string = iso_string[1:]
        if 'H' in iso_string:
            hour_part, iso_string = iso_string.split('H', 1)
            hours = int(hour_part)
        if 'M' in iso_string:
            minute_part, iso_string = iso_string.split('M', 1)
            minutes = int(minute_part)
        if 'S' in iso_string:
            second_part = iso_string.split('S')[0]
            seconds = float(second_part)

    total_seconds = days * 86400 + hours * 3600 + minutes * 60 + seconds
    if is_negative:
        total_seconds = -total_seconds

    return total_seconds


@given(st.integers(min_value=-1000000, max_value=1000000))
def test_timedelta_isoformat_semantic_correctness(seconds):
    """Property: ISO format output should represent the same duration as the input."""
    td = datetime.timedelta(seconds=seconds)
    iso_output = timedelta_isoformat(td)

    expected_seconds = td.total_seconds()
    parsed_seconds = parse_iso_duration_to_seconds(iso_output)

    assert abs(expected_seconds - parsed_seconds) < 0.001, \
        f"ISO format {iso_output} represents {parsed_seconds}s but should represent {expected_seconds}s"
```

**Failing input**: `seconds=-1`

## Reproducing the Bug

```python
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import datetime
from pydantic.deprecated.json import timedelta_isoformat

td = datetime.timedelta(seconds=-1)

print(f"Input: timedelta(seconds=-1)")
print(f"Python internal representation: {td}")
print(f"Actual duration (total_seconds): {td.total_seconds()}")

iso_output = timedelta_isoformat(td)
print(f"ISO format output: {iso_output}")

print("\nProblem:")
print(f"  The ISO format '-P1DT23H59M59.000000S' means:")
print(f"  'negative (1 day + 23 hours + 59 minutes + 59 seconds)'")
print(f"  = -(86400 + 82800 + 3540 + 59) = -172799 seconds")
print(f"  But the actual timedelta is -1 second!")
print(f"  The correct ISO format should be: -PT1S")
```

Output:
```
Input: timedelta(seconds=-1)
Python internal representation: -1 day, 23:59:59
Actual duration (total_seconds): -1.0
ISO format output: -P1DT23H59M59.000000S

Problem:
  The ISO format '-P1DT23H59M59.000000S' means:
  'negative (1 day + 23 hours + 59 minutes + 59 seconds)'
  = -(86400 + 82800 + 3540 + 59) = -172799 seconds
  But the actual timedelta is -1 second!
  The correct ISO format should be: -PT1S
```

## Why This Is A Bug

Python represents negative timedeltas using negative days and positive seconds/microseconds. For example, `timedelta(seconds=-1)` is internally stored as `timedelta(days=-1, seconds=86399, microseconds=0)`.

The current implementation formats this internal representation directly:

```python
def timedelta_isoformat(td: datetime.timedelta) -> str:
    minutes, seconds = divmod(td.seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f'{"-" if td.days < 0 else ""}P{abs(td.days)}DT{hours:d}H{minutes:d}M{seconds:d}.{td.microseconds:06d}S'
```

For `timedelta(seconds=-1)`:
- `td.days = -1`, `td.seconds = 86399`, `td.microseconds = 0`
- Output: `-P1DT23H59M59.000000S`
- Semantic meaning: "negative (1 day + 23:59:59)" = -172799 seconds
- Actual value: -1 second

This creates a massive discrepancy (172798 seconds difference) and violates the ISO 8601 standard's intent.

## Fix

```diff
def timedelta_isoformat(td: datetime.timedelta) -> str:
-   minutes, seconds = divmod(td.seconds, 60)
-   hours, minutes = divmod(minutes, 60)
-   return f'{"-" if td.days < 0 else ""}P{abs(td.days)}DT{hours:d}H{minutes:d}M{seconds:d}.{td.microseconds:06d}S'
+   total_seconds = td.total_seconds()
+   is_negative = total_seconds < 0
+   if is_negative:
+       total_seconds = -total_seconds
+
+   microseconds = int((total_seconds % 1) * 1_000_000)
+   total_seconds = int(total_seconds)
+
+   days, remainder = divmod(total_seconds, 86400)
+   hours, remainder = divmod(remainder, 3600)
+   minutes, seconds = divmod(remainder, 60)
+
+   return f'{"-" if is_negative else ""}P{days}DT{hours}H{minutes}M{seconds}.{microseconds:06d}S'
```