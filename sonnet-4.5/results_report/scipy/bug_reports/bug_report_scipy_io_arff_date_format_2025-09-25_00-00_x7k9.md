# Bug Report: scipy.io.arff DateAttribute Invalid Format Validation

**Target**: `scipy.io.arff._arffread.DateAttribute._get_date_format`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `DateAttribute._get_date_format` method fails to raise `ValueError` for invalid or empty date format patterns due to a logic error on line 276 where `elif "yy":` is always `True`, causing it to incorrectly set `datetime_unit = "Y"` even when the pattern contains no valid date components.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from scipy.io.arff._arffread import DateAttribute
import pytest


@given(st.text(alphabet=' \t', min_size=0, max_size=10))
def test_date_format_empty_or_whitespace_bug(whitespace):
    pattern = f"date '{whitespace}'"
    try:
        attr = DateAttribute.parse_attribute('test', pattern)
        pytest.fail(f"Should have raised ValueError for pattern '{pattern}', "
                   f"but got datetime_unit={attr.datetime_unit}")
    except ValueError:
        pass
```

**Failing input**: `whitespace=''` (or any whitespace-only string)

## Reproducing the Bug

```python
from scipy.io.arff._arffread import DateAttribute

attr = DateAttribute.parse_attribute('test_attr', "date ''")
print(f"Empty date format should raise ValueError")
print(f"But got: datetime_unit={attr.datetime_unit}, date_format='{attr.date_format}'")
```

## Why This Is A Bug

According to the code's intent (lines 298-299), if `datetime_unit` is `None` after checking all pattern components, a `ValueError` should be raised for "Invalid or unsupported date format". However, the buggy condition `elif "yy":` on line 276 is always `True` (since `"yy"` is a non-empty string), causing `datetime_unit` to be set to `"Y"` unconditionally for any pattern that doesn't contain `"yyyy"`. This prevents the `ValueError` from being raised for genuinely invalid patterns like empty strings or whitespace-only formats.

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