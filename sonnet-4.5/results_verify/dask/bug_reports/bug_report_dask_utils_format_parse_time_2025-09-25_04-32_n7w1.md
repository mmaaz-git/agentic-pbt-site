# Bug Report: dask.utils - format_time and parse_timedelta Incompatibility

**Target**: `dask.utils.format_time` and `dask.utils.parse_timedelta`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `format_time` function produces multi-unit output strings (e.g., "10m 1s", "24hr 0m") that cannot be parsed by `parse_timedelta`, breaking the expected round-trip property for format/parse function pairs.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from dask.utils import format_time, parse_timedelta

@given(st.floats(min_value=1e-6, max_value=1e8, allow_nan=False, allow_infinity=False))
@settings(max_examples=300)
def test_format_parse_time_roundtrip(t):
    """Property: format_time produces parseable output"""
    formatted = format_time(t)
    parsed = parse_timedelta(formatted)
    assert abs(parsed - t) / t < 0.05
```

**Failing input**: `t=601.0` (and any value >= 600 that produces multi-unit format)

## Reproducing the Bug

```python
from dask.utils import format_time, parse_timedelta

test_values = [601, 3600, 7200, 86400]

for t in test_values:
    formatted = format_time(t)
    print(f"format_time({t}) = '{formatted}'")
    try:
        parsed = parse_timedelta(formatted)
        print(f"  Parsed successfully: {parsed}")
    except ValueError as e:
        print(f"  ERROR: {e}")
```

Output:
```
format_time(601) = '10m 1s'
  ERROR: could not convert string to float: '10m1'
format_time(3600) = '60m 0s'
  ERROR: could not convert string to float: '60m0'
format_time(7200) = '120m 0s'
  ERROR: could not convert string to float: '120m0'
format_time(86400) = '24hr 0m'
  ERROR: could not convert string to float: '24hr0'
```

## Why This Is A Bug

1. **Complementary function names** suggest they should work together
2. **Same module** (dask.utils) implies related functionality
3. **Natural expectation**: `parse_timedelta(format_time(x))` should approximately equal `x`
4. **Inconsistent design**: Some format_time outputs parse successfully (single-unit), others don't (multi-unit)

The issue is in `dask/utils.py`:
- `format_time` (lines 1661-1699) produces multi-unit strings like "10m 1s"
- `parse_timedelta` (lines 1846-1900) only handles single-unit strings

When `parse_timedelta` tries to parse "10m 1s", it:
1. Removes spaces: "10m1s"
2. Finds the last non-alpha character: 'm' at position 2
3. Tries to parse prefix "10m1" as a float, which fails

## Fix

Option 1: Modify `parse_timedelta` to handle multi-unit strings:

```diff
def parse_timedelta(s, default="seconds"):
    ...
    s = s.replace(" ", "")
    if not s[0].isdigit():
        s = "1" + s

+   # Handle multi-unit strings like "10m1s", "24hr0m"
+   import re
+   multi_unit_pattern = r'(\d+(?:\.\d+)?)(hr|m|s|ms|us|d|w)'
+   matches = re.findall(multi_unit_pattern, s)
+   if len(matches) > 1:
+       total = 0
+       for value, unit in matches:
+           total += float(value) * timedelta_sizes[unit]
+       return int(total) if int(total) == total else total

    for i in range(len(s) - 1, -1, -1):
        if not s[i].isalpha():
            break
    ...
```

Option 2: Modify `format_time` to produce single-unit output:

```diff
def format_time(n: float) -> str:
    """format integers as time
    ...
    """
    if n > 24 * 60 * 60 * 2:
-       d = int(n / 3600 / 24)
-       h = int((n - d * 3600 * 24) / 3600)
-       return f"{d}d {h}hr"
+       d = n / 3600 / 24
+       return f"{d:.2f}d"
    if n > 60 * 60 * 2:
-       h = int(n / 3600)
-       m = int((n - h * 3600) / 60)
-       return f"{h}hr {m}m"
+       h = n / 3600
+       return f"{h:.2f}hr"
    if n > 60 * 10:
-       m = int(n / 60)
-       s = int(n - m * 60)
-       return f"{m}m {s}s"
+       m = n / 60
+       return f"{m:.2f}m"
    ...
```

Option 1 is preferable as it extends `parse_timedelta` to handle more formats without breaking existing functionality.