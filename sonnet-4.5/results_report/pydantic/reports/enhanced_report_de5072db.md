# Bug Report: pydantic.deprecated.json.timedelta_isoformat Incorrect ISO 8601 Encoding for Negative Timedeltas

**Target**: `pydantic.deprecated.json.timedelta_isoformat`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `timedelta_isoformat` function produces incorrect ISO 8601 duration strings for negative timedeltas where the internal representation has `days < 0` but `seconds > 0` or `microseconds > 0`. This results in duration strings that mathematically represent different values than the original timedelta.

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


if __name__ == "__main__":
    test_timedelta_isoformat_roundtrip()
```

<details>

<summary>
**Failing input**: `timedelta(days=-1, microseconds=1)`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/1/hypo.py", line 72, in <module>
    test_timedelta_isoformat_roundtrip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/1/hypo.py", line 57, in test_timedelta_isoformat_roundtrip
    @given(st.timedeltas(min_value=timedelta(days=-365), max_value=timedelta(days=365)))
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/1/hypo.py", line 68, in test_timedelta_isoformat_roundtrip
    assert abs(reconstructed_seconds - original_seconds) < 0.000001
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_timedelta_isoformat_roundtrip(
    td=datetime.timedelta(days=-1, microseconds=1),
)
```
</details>

## Reproducing the Bug

```python
import warnings
from datetime import timedelta
from pydantic.deprecated.json import timedelta_isoformat

# Test case: negative timedelta with microseconds
td = timedelta(days=-1, microseconds=1)

print(f"timedelta: {td}")
print(f"  days={td.days}, seconds={td.seconds}, microseconds={td.microseconds}")
print(f"  total_seconds()={td.total_seconds()}")

# Get the ISO format
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    iso = timedelta_isoformat(td)

print(f"  ISO format: {iso}")
print(f"\nInterpretation of ISO format '{iso}':")
print(f"  According to ISO 8601, '-P1DT0H0M0.000001S' means:")
print(f"    -(1 day + 0 hours + 0 minutes + 0.000001 seconds)")
print(f"    = -(86400 + 0.000001) seconds")
print(f"    = -86400.000001 seconds")
print(f"  But the actual timedelta represents: {td.total_seconds()} seconds")
print(f"  Difference: {abs(-86400.000001 - td.total_seconds())} seconds")
print("\nThe bug: The function incorrectly encodes this timedelta.")
print("The ISO string suggests -86400.000001 seconds but the timedelta is -86399.999999 seconds.")
```

<details>

<summary>
Output showing the encoding mismatch
</summary>
```
timedelta: -1 day, 0:00:00.000001
  days=-1, seconds=0, microseconds=1
  total_seconds()=-86399.999999
  ISO format: -P1DT0H0M0.000001S

Interpretation of ISO format '-P1DT0H0M0.000001S':
  According to ISO 8601, '-P1DT0H0M0.000001S' means:
    -(1 day + 0 hours + 0 minutes + 0.000001 seconds)
    = -(86400 + 0.000001) seconds
    = -86400.000001 seconds
  But the actual timedelta represents: -86399.999999 seconds
  Difference: 1.9999861251562834e-06 seconds

The bug: The function incorrectly encodes this timedelta.
The ISO string suggests -86400.000001 seconds but the timedelta is -86399.999999 seconds.
```
</details>

## Why This Is A Bug

Python's `timedelta` internally normalizes its values so that `seconds` is always in the range [0, 86399] and `microseconds` is always in the range [0, 999999], with only `days` carrying the negative sign. For example, `timedelta(days=-1, microseconds=1)` actually represents -86399.999999 seconds, not -86400.000001 seconds.

The current implementation at line 141 of `/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages/pydantic/deprecated/json.py` incorrectly handles this normalization:

```python
return f'{"-" if td.days < 0 else ""}P{abs(td.days)}DT{hours:d}H{minutes:d}M{seconds:d}.{td.microseconds:06d}S'
```

This code simply prepends a minus sign when `days < 0` and then uses the absolute value of days along with the already-normalized positive `seconds` and `microseconds` values. According to the ISO 8601 standard, when a minus sign precedes the 'P', it applies to the entire duration sum. Therefore, `-P1DT0H0M0.000001S` means "negative (1 day + 0.000001 seconds)" which equals -86400.000001 seconds, not the -86399.999999 seconds that the original timedelta represents.

## Relevant Context

The ISO 8601 standard specifies that durations are represented in the format `P[n]Y[n]M[n]DT[n]H[n]M[n]S` where:
- P is the duration designator (period)
- T is the time designator that precedes the time components
- A negative duration has a minus sign before the P

When a negative sign is present, it applies to the entire duration. All individual components should then be positive values that are summed together before applying the negative sign.

The function is located in the deprecated module (`pydantic.deprecated.json`) and is marked with deprecation warnings, indicating that users should migrate to the modern Pydantic v2 approach using `ConfigDict(ser_json_timedelta='iso8601')`. However, as long as the function exists and claims to provide ISO 8601 encoding, it should do so correctly.

## Proposed Fix

```diff
--- a/pydantic/deprecated/json.py
+++ b/pydantic/deprecated/json.py
@@ -136,6 +136,17 @@ def timedelta_isoformat(td: datetime.timedelta) -> str:
 def timedelta_isoformat(td: datetime.timedelta) -> str:
     """ISO 8601 encoding for Python timedelta object."""
     warnings.warn('`timedelta_isoformat` is deprecated.', category=PydanticDeprecatedSince20, stacklevel=2)
-    minutes, seconds = divmod(td.seconds, 60)
-    hours, minutes = divmod(minutes, 60)
-    return f'{"-" if td.days < 0 else ""}P{abs(td.days)}DT{hours:d}H{minutes:d}M{seconds:d}.{td.microseconds:06d}S'
+
+    # Calculate total duration in seconds to handle negative timedeltas correctly
+    total_seconds = int(td.total_seconds())
+    is_negative = total_seconds < 0
+    total_seconds = abs(total_seconds)
+
+    # Break down into components
+    days, remainder = divmod(total_seconds, 86400)
+    hours, remainder = divmod(remainder, 3600)
+    minutes, seconds = divmod(remainder, 60)
+
+    # Handle microseconds separately to preserve precision
+    microseconds = abs(td.total_seconds() * 1000000) % 1000000
+    return f'{"-" if is_negative else ""}P{days}DT{hours}H{minutes}M{seconds}.{int(microseconds):06d}S'
```