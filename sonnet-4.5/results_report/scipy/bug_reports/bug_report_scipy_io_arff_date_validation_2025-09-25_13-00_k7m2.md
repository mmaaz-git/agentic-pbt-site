# Bug Report: scipy.io.arff DateAttribute Invalid Pattern Validation

**Target**: `scipy.io.arff._arffread.DateAttribute._get_date_format`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `DateAttribute._get_date_format` method incorrectly accepts invalid date format patterns due to a logic error on line 276 where `elif "yy":` should be `elif "yy" in pattern:`. This causes the function to fail to validate date formats properly, accepting patterns that contain no valid date components.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from scipy.io.arff._arffread import DateAttribute
import pytest


@given(st.text(alphabet='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ', min_size=1, max_size=10))
def test_invalid_date_format_should_raise_valueerror(invalid_pattern):
    assume('yyyy' not in invalid_pattern)
    assume('yy' not in invalid_pattern)
    assume('MM' not in invalid_pattern)
    assume('dd' not in invalid_pattern)
    assume('HH' not in invalid_pattern)
    assume('mm' not in invalid_pattern)
    assume('ss' not in invalid_pattern)

    date_string = f'date "{invalid_pattern}"'

    with pytest.raises(ValueError):
        DateAttribute._get_date_format(date_string)
```

**Failing input**: `'date "abc"'` (or any pattern without valid date components)

## Reproducing the Bug

```python
from scipy.io.arff._arffread import DateAttribute

invalid_pattern = 'date "abc"'

try:
    result_pattern, result_unit = DateAttribute._get_date_format(invalid_pattern)
    print(f"BUG: Invalid pattern '{invalid_pattern}' was accepted")
    print(f"Returned: pattern='{result_pattern}', unit='{result_unit}'")
    print(f"Expected: Should raise ValueError")
except ValueError as e:
    print(f"Correct: ValueError raised - {e}")
```

## Why This Is A Bug

The function is designed to validate and convert Java SimpleDateFormat patterns to Python datetime patterns. When given an invalid pattern containing no recognized date components (like "abc"), it should raise a ValueError indicating the pattern is invalid.

However, due to the bug on line 276, the condition `elif "yy":` always evaluates to `True` (since the string `"yy"` is truthy), causing the function to set `datetime_unit = "Y"` even when there's no year component in the pattern. This prevents the validation check on line 298 from catching invalid patterns.

## Fix

```diff
--- a/scipy/io/arff/_arffread.py
+++ b/scipy/io/arff/_arffread.py
@@ -273,7 +273,7 @@ class DateAttribute(Attribute):
         if "yyyy" in pattern:
             pattern = pattern.replace("yyyy", "%Y")
             datetime_unit = "Y"
-        elif "yy":
+        elif "yy" in pattern:
             pattern = pattern.replace("yy", "%y")
             datetime_unit = "Y"
         if "MM" in pattern:
```