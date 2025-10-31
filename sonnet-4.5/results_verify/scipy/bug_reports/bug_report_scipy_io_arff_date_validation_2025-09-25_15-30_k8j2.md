# Bug Report: scipy.io.arff DateAttribute Invalid Pattern Validation

**Target**: `scipy.io.arff._arffread.DateAttribute._get_date_format`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

DateAttribute._get_date_format fails to raise ValueError for invalid date patterns due to a logic error on line 276. The condition `elif "yy":` should be `elif "yy" in pattern:`, causing the elif branch to always execute and incorrectly set datetime_unit='Y' even for patterns with no valid date components.

## Property-Based Test

```python
from scipy.io.arff._arffread import DateAttribute
import pytest


def test_invalid_date_format_should_fail():
    invalid_pattern = "date abc"

    with pytest.raises(ValueError, match="Invalid or unsupported date format"):
        DateAttribute.parse_attribute("test", invalid_pattern)
```

**Failing input**: `pattern = "date abc"`

## Reproducing the Bug

```python
from scipy.io.arff._arffread import DateAttribute

invalid_pattern = "date abc"
attr = DateAttribute.parse_attribute("test", invalid_pattern)

print(f"date_format: {attr.date_format}")
print(f"datetime_unit: {attr.datetime_unit}")
```

Expected: Raises `ValueError("Invalid or unsupported date format")`
Actual: Returns DateAttribute with date_format='abc' and datetime_unit='Y'

## Why This Is A Bug

The code at line 276 has:
```python
elif "yy":
```

This should be:
```python
elif "yy" in pattern:
```

The string literal `"yy"` is always truthy, so the elif branch executes whenever the `if "yyyy" in pattern:` condition is false. This causes two problems:

1. **Bypasses validation**: The `datetime_unit` variable gets set to 'Y' even for patterns with no valid date components (like "date abc"). This prevents the validation check at line 298-299 from raising the expected ValueError.

2. **Incorrect replacement**: Line 277 calls `pattern.replace("yy", "%y")` on patterns that don't contain "yy", which is wasteful (though harmless).

The code correctly uses `if "..." in pattern:` for all other date components (yyyy, MM, dd, HH, mm, ss, z, Z), making line 276 an obvious oversight.

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