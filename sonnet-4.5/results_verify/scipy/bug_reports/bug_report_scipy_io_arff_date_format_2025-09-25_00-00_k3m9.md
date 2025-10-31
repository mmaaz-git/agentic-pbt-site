# Bug Report: scipy.io.arff DateAttribute._get_date_format Invalid Pattern Handling

**Target**: `scipy.io.arff._arffread.DateAttribute._get_date_format`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `DateAttribute._get_date_format` method incorrectly accepts invalid date patterns and returns incorrect datetime units instead of raising a ValueError. This is caused by a logic error on line 276 where `elif "yy":` is always True.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from scipy.io.arff._arffread import DateAttribute
import pytest


@given(st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll')), min_size=1, max_size=20))
def test_date_format_invalid_patterns_should_raise(pattern):
    valid_components = ['yyyy', 'yy', 'MM', 'dd', 'HH', 'mm', 'ss']
    assume(not any(comp in pattern for comp in valid_components))
    assume('z' not in pattern.lower() and 'Z' not in pattern)

    try:
        result_pattern, unit = DateAttribute._get_date_format(f"date {pattern}")

        pytest.fail(
            f"Pattern '{pattern}' has no valid date components but returned "
            f"result='{result_pattern}', unit='{unit}' instead of raising ValueError. "
            f"This is due to bug on line 276: 'elif \"yy\":' which is always True"
        )
    except ValueError:
        pass
```

**Failing input**: `pattern='A'` (or any string without valid date components)

## Reproducing the Bug

```python
from scipy.io.arff._arffread import DateAttribute

pattern = "A"
result_pattern, unit = DateAttribute._get_date_format(f"date {pattern}")

print(f"Pattern: {pattern}")
print(f"Result: {result_pattern}")
print(f"Unit: {unit}")
```

Output:
```
Pattern: A
Result: A
Unit: Y
```

Expected: Should raise `ValueError("Invalid or unsupported date format")`

## Why This Is A Bug

The bug is on line 276 of `_arffread.py`:

```python
if "yyyy" in pattern:
    pattern = pattern.replace("yyyy", "%Y")
    datetime_unit = "Y"
elif "yy":  # BUG: This is always True!
    pattern = pattern.replace("yy", "%y")
    datetime_unit = "Y"
```

The condition `elif "yy":` checks the truthiness of the string literal `"yy"`, which is always True for non-empty strings. This should be `elif "yy" in pattern:` to check if "yy" exists in the pattern.

As a result:
1. Invalid patterns that don't contain any date components are accepted
2. The datetime_unit is set to 'Y' (year) even when the pattern has no year component
3. Only patterns that have MM, dd, HH, mm, or ss components "hide" this bug because they overwrite datetime_unit later

The code should validate that at least one valid date component exists and raise a ValueError if none are found. Currently, the check on lines 298-299 only triggers if datetime_unit is None, but due to the bug, it's always set to 'Y' even for completely invalid patterns.

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