# Bug Report: scipy.io.arff DateAttribute._get_date_format Always True Condition

**Target**: `scipy.io.arff._arffread.DateAttribute._get_date_format`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `DateAttribute._get_date_format` method has an always-true condition `elif "yy":` on line 276, which should be `elif "yy" in pattern:`. This causes the method to incorrectly overwrite the `datetime_unit` variable for date formats that don't contain "yy" or "yyyy".

## Property-Based Test

```python
from hypothesis import given, strategies as st
from scipy.io.arff._arffread import DateAttribute


@given(st.sampled_from([
    "date 'MM'",
    "date 'dd'",
    "date 'HH'",
    "date 'mm'",
    "date 'ss'",
    "date 'MM-dd'",
    "date 'HH:mm:ss'"
]))
def test_date_format_unit_matches_most_precise_component(date_string):
    """
    Test that the datetime_unit matches the most precise time component in the format.
    For example, 'MM' alone should have unit 'M', not 'Y'.
    """
    date_format, datetime_unit = DateAttribute._get_date_format(date_string)

    pattern = date_string.split("'")[1]

    if 'ss' in pattern:
        assert datetime_unit == 's', f"Expected unit 's' for {pattern}, got {datetime_unit}"
    elif 'mm' in pattern:
        assert datetime_unit == 'm', f"Expected unit 'm' for {pattern}, got {datetime_unit}"
    elif 'HH' in pattern:
        assert datetime_unit == 'h', f"Expected unit 'h' for {pattern}, got {datetime_unit}"
    elif 'dd' in pattern:
        assert datetime_unit == 'D', f"Expected unit 'D' for {pattern}, got {datetime_unit}"
    elif 'MM' in pattern:
        assert datetime_unit == 'M', f"Expected unit 'M' for {pattern}, got {datetime_unit}"
```

**Failing input**: `"date 'MM'"` (and any other format without "yyyy" or "yy")

## Reproducing the Bug

```python
from scipy.io.arff._arffread import DateAttribute

date_format, datetime_unit = DateAttribute._get_date_format("date 'MM'")
print(f"Format: {date_format}, Unit: {datetime_unit}")

assert datetime_unit == "M", f"Expected unit 'M' for month-only format, got '{datetime_unit}'"
```

Expected output: `Format: %m, Unit: M`
Actual output: `Format: %m-%m, Unit: Y`

The unit is incorrectly set to 'Y' instead of 'M'.

## Why This Is A Bug

The code at lines 273-278 in `_arffread.py` contains a logic error:

```python
if "yyyy" in pattern:
    pattern = pattern.replace("yyyy", "%Y")
    datetime_unit = "Y"
elif "yy":
    pattern = pattern.replace("yy", "%y")
    datetime_unit = "Y"
```

The condition `elif "yy":` is always true because "yy" is a non-empty string, which is truthy in Python. This means that whenever the pattern doesn't contain "yyyy", the code will always execute the `elif` branch and:

1. Replace "yy" in the pattern (even if it's not present), potentially causing double replacements
2. Overwrite `datetime_unit` to "Y" even when the pattern doesn't contain any year component

This breaks the logic where `datetime_unit` should be set to the most precise time component in the format string (seconds > minutes > hours > days > months > years).

For example, when processing `"date 'MM'"`:
- Line 279-280 correctly sets `datetime_unit = "M"` when it finds "MM"
- Then line 276-278 overwrites it to `datetime_unit = "Y"` because the `elif "yy":` condition is always true

This affects any date format that doesn't include "yyyy" but includes other components like MM, dd, HH, mm, or ss.

## Fix

```diff
--- a/scipy/io/arff/_arffread.py
+++ b/scipy/io/arff/_arffread.py
@@ -273,7 +273,7 @@ class DateAttribute(Attribute):
             if "yyyy" in pattern:
                 pattern = pattern.replace("yyyy", "%Y")
                 datetime_unit = "Y"
-            elif "yy":
+            elif "yy" in pattern:
                 pattern = pattern.replace("yy", "%y")
                 datetime_unit = "Y"
             if "MM" in pattern:
```