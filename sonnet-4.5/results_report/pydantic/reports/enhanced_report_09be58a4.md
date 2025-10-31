# Bug Report: pydantic.deprecated.json.timedelta_isoformat Produces Incorrect ISO 8601 Duration for Negative Timedeltas with Microseconds

**Target**: `pydantic.deprecated.json.timedelta_isoformat`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `timedelta_isoformat` function incorrectly encodes negative timedeltas that have non-zero seconds or microseconds components, producing ISO 8601 duration strings that represent a different time value than the original timedelta.

## Property-Based Test

```python
import datetime
import re
from hypothesis import given, strategies as st
from pydantic.deprecated.json import timedelta_isoformat


def parse_iso_duration(iso_string):
    match = re.match(r'^(-?)P(\d+)DT(\d+)H(\d+)M(\d+)\.(\d+)S$', iso_string)
    if not match:
        raise ValueError(f"Invalid ISO duration: {iso_string}")

    negative = match.group(1) == '-'
    days = int(match.group(2))
    hours = int(match.group(3))
    minutes = int(match.group(4))
    seconds = int(match.group(5))
    microseconds = int(match.group(6))

    total_seconds = days * 86400 + hours * 3600 + minutes * 60 + seconds + microseconds / 1_000_000

    if negative:
        total_seconds = -total_seconds

    return total_seconds


@given(st.timedeltas(
    min_value=datetime.timedelta(days=-999999),
    max_value=datetime.timedelta(days=999999)
))
def test_timedelta_isoformat_roundtrip_value(td):
    iso_str = timedelta_isoformat(td)

    reconstructed_seconds = parse_iso_duration(iso_str)
    original_seconds = td.total_seconds()

    assert abs(reconstructed_seconds - original_seconds) < 1e-6, (
        f"ISO format doesn't preserve value!\n"
        f"  Original: {td} ({original_seconds} seconds)\n"
        f"  ISO: {iso_str}\n"
        f"  Reconstructed: {reconstructed_seconds} seconds\n"
        f"  Difference: {reconstructed_seconds - original_seconds} seconds"
    )


if __name__ == "__main__":
    test_timedelta_isoformat_roundtrip_value()
```

<details>

<summary>
**Failing input**: `datetime.timedelta(days=-1, microseconds=1)`
</summary>
```
/home/npc/pbt/agentic-pbt/worker_/29/hypo.py:32: PydanticDeprecatedSince20: `timedelta_isoformat` is deprecated. Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.10/migration/
  iso_str = timedelta_isoformat(td)
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/29/hypo.py", line 47, in <module>
    test_timedelta_isoformat_roundtrip_value()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/29/hypo.py", line 28, in test_timedelta_isoformat_roundtrip_value
    min_value=datetime.timedelta(days=-999999),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/29/hypo.py", line 37, in test_timedelta_isoformat_roundtrip_value
    assert abs(reconstructed_seconds - original_seconds) < 1e-6, (
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: ISO format doesn't preserve value!
  Original: -1 day, 0:00:00.000001 (-86399.999999 seconds)
  ISO: -P1DT0H0M0.000001S
  Reconstructed: -86400.000001 seconds
  Difference: -1.9999861251562834e-06 seconds
Falsifying example: test_timedelta_isoformat_roundtrip_value(
    td=datetime.timedelta(days=-1, microseconds=1),
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/29/hypo.py:38
```
</details>

## Reproducing the Bug

```python
import datetime
from pydantic.deprecated.json import timedelta_isoformat

td = datetime.timedelta(days=-1, microseconds=1)

print(f"Timedelta: {td}")
print(f"Total seconds: {td.total_seconds()}")
print(f"Days component: {td.days}")
print(f"Seconds component: {td.seconds}")
print(f"Microseconds component: {td.microseconds}")

iso_str = timedelta_isoformat(td)
print(f"\nISO format: {iso_str}")

print(f"\nExpected total seconds: {td.total_seconds()}")
print(f"ISO string represents: -(1 day + 0 hours + 0 minutes + 0.000001 seconds)")
print(f"Which equals: -86400.000001 seconds")
print(f"Difference from expected: {-86400.000001 - td.total_seconds()} seconds")
```

<details>

<summary>
Output showing incorrect ISO 8601 encoding
</summary>
```
/home/npc/pbt/agentic-pbt/worker_/29/repo.py:12: PydanticDeprecatedSince20: `timedelta_isoformat` is deprecated. Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.10/migration/
  iso_str = timedelta_isoformat(td)
Timedelta: -1 day, 0:00:00.000001
Total seconds: -86399.999999
Days component: -1
Seconds component: 0
Microseconds component: 1

ISO format: -P1DT0H0M0.000001S

Expected total seconds: -86399.999999
ISO string represents: -(1 day + 0 hours + 0 minutes + 0.000001 seconds)
Which equals: -86400.000001 seconds
Difference from expected: -1.9999861251562834e-06 seconds
```
</details>

## Why This Is A Bug

This violates the ISO 8601 standard's expected behavior for duration encoding. The bug stems from a fundamental misunderstanding of Python's internal timedelta representation:

1. **Python's timedelta internal representation**: For negative timedeltas, Python internally normalizes the value so that `days` is negative while `seconds` and `microseconds` are always non-negative. For example, `timedelta(days=-1, microseconds=1)` is stored as:
   - `days = -1`
   - `seconds = 0`
   - `microseconds = 1`

   This represents: -1 day + 0 seconds + 1 microsecond = -86399.999999 seconds total

2. **The bug**: The current implementation in lines 139-141 of `/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages/pydantic/deprecated/json.py` naively uses these internal components:
   ```python
   minutes, seconds = divmod(td.seconds, 60)
   hours, minutes = divmod(minutes, 60)
   return f'{"-" if td.days < 0 else ""}P{abs(td.days)}DT{hours:d}H{minutes:d}M{seconds:d}.{td.microseconds:06d}S'
   ```

   This produces `-P1DT0H0M0.000001S` which, according to ISO 8601, means "negative (1 day + 0.000001 seconds)" = -86400.000001 seconds.

3. **The discrepancy**: The function produces an ISO string representing -86400.000001 seconds when the original timedelta represents -86399.999999 seconds - a difference of approximately 2 microseconds.

## Relevant Context

- The function is already deprecated in Pydantic V2.0 and scheduled for removal in V3.0
- The ISO 8601 standard defines that a negative sign before a duration applies to the entire duration value
- Python's documentation for timedelta states that only the `days` attribute can be negative, while `seconds` and `microseconds` are always in the range [0, 86400) and [0, 1000000) respectively
- The recommended migration path is to use `pydantic_core.to_jsonable_python` instead
- This bug affects any negative timedelta where the time portion (seconds + microseconds) is non-zero

## Proposed Fix

```diff
 def timedelta_isoformat(td: datetime.timedelta) -> str:
     """ISO 8601 encoding for Python timedelta object."""
     warnings.warn('`timedelta_isoformat` is deprecated.', category=PydanticDeprecatedSince20, stacklevel=2)
-    minutes, seconds = divmod(td.seconds, 60)
-    hours, minutes = divmod(minutes, 60)
-    return f'{"-" if td.days < 0 else ""}P{abs(td.days)}DT{hours:d}H{minutes:d}M{seconds:d}.{td.microseconds:06d}S'
+    total_seconds = td.total_seconds()
+    negative = total_seconds < 0
+
+    if negative:
+        td = -td
+
+    days = td.days
+    hours, remainder = divmod(td.seconds, 3600)
+    minutes, seconds = divmod(remainder, 60)
+    microseconds = td.microseconds
+
+    return f'{"-" if negative else ""}P{days}DT{hours}H{minutes}M{seconds}.{microseconds:06d}S'
```