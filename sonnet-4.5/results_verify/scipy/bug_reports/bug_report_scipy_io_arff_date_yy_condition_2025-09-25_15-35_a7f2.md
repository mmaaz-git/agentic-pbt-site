# Bug Report: scipy.io.arff DateAttribute Incorrect Condition for 'yy'

**Target**: `scipy.io.arff._arffread.DateAttribute._get_date_format`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_get_date_format` method has a logic error at line 276 where it checks `elif "yy":` instead of `elif "yy" in pattern:`. This causes the elif block to always execute when the pattern doesn't contain "yyyy", even when "yy" is not present in the pattern. This results in incorrect `datetime_unit` assignment for date patterns with no date/time components.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
from scipy.io.arff._arffread import DateAttribute


@given(st.text().filter(lambda x: 'yyyy' not in x.lower() and 'yy' not in x.lower()
                                     and 'mm' not in x.lower() and 'dd' not in x.lower()
                                     and 'hh' not in x.lower() and 'ss' not in x.lower()))
@settings(max_examples=100)
def test_date_format_no_components_should_fail(text):
    assume(len(text.strip()) > 0)

    try:
        pattern, unit = DateAttribute._get_date_format(f"date {text}")
        assert False, f"Should have raised ValueError, but got unit={unit}"
    except ValueError:
        pass
```

**Failing input**: `"date 0"` (and many other patterns without actual date components)

## Reproducing the Bug

```python
from scipy.io.arff._arffread import DateAttribute

pattern, unit = DateAttribute._get_date_format("date 'just text'")
print(f"Pattern: {pattern}, Unit: {unit}")
```

Output:
```
Pattern: just text, Unit: Y
```

Expected: Should raise `ValueError` because the pattern contains no actual date/time format components.

## Why This Is A Bug

At line 276, the code has:
```python
elif "yy":
    pattern = pattern.replace("yy", "%y")
    datetime_unit = "Y"
```

The condition `elif "yy":` always evaluates to `True` because `"yy"` is a non-empty string. This should be `elif "yy" in pattern:`.

The bug causes patterns with no date/time components to incorrectly succeed with `datetime_unit="Y"` instead of raising a ValueError. While most real-world date patterns include actual components, this logic error could lead to subtle bugs and violates the function's contract.

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