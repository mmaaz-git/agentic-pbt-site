# Bug Report: scipy.io.arff Date Format Validation Bypass

**Target**: `scipy.io.arff._arffread.DateAttribute._get_date_format`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `DateAttribute._get_date_format` method contains a logic error on line 276 where `elif "yy":` should be `elif "yy" in pattern:`. This causes the method to accept invalid date format patterns that contain no valid date/time components, bypassing the intended validation.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, example
from scipy.io.arff._arffread import DateAttribute
import pytest


@given(st.text(
    alphabet=st.characters(
        blacklist_characters='yMdHmsYzZ0123456789',
        blacklist_categories=('Cs', 'Cc')
    ),
    min_size=1,
    max_size=30
).filter(lambda s: s.strip()))
@settings(max_examples=500)
@example("foobar")
@example("xyz")
@example("invalid")
def test_invalid_date_format_rejected(invalid_pattern):
    """
    Property: Date patterns without any valid date components should raise ValueError.
    Valid components are: yyyy, yy, MM, dd, HH, mm, ss
    """
    attr_string = f'date "{invalid_pattern}"'

    with pytest.raises(ValueError, match="Invalid"):
        DateAttribute._get_date_format(attr_string)
```

**Failing input**: `"foobar"`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/scipy_env/lib/python3.13/site-packages')

from scipy.io.arff._arffread import DateAttribute

attr_string = 'date "foobar"'
date_format, datetime_unit = DateAttribute._get_date_format(attr_string)

print(f"Pattern: foobar")
print(f"Returned format: {date_format}")
print(f"Returned unit: {datetime_unit}")
```

Expected: `ValueError: Invalid or unsupported date format`
Actual: Returns `date_format="foobar"`, `datetime_unit="Y"` (invalid result)

## Why This Is A Bug

The condition on line 276 of `_arffread.py`:
```python
elif "yy":
```

checks if the string literal `"yy"` is truthy (which is always True), rather than checking if `"yy"` exists in the pattern. This causes the `elif` branch to execute for ANY pattern that doesn't contain `"yyyy"`, including completely invalid patterns like `"foobar"`.

When this happens:
1. `datetime_unit` is set to `"Y"`
2. The subsequent validation check `if datetime_unit is None:` on line 298 passes
3. Invalid patterns are accepted instead of raising a `ValueError`

The code should check `elif "yy" in pattern:` to properly validate that the pattern contains the "yy" component before processing it.

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