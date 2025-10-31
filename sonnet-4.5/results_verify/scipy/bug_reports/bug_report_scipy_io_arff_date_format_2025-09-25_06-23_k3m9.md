# Bug Report: scipy.io.arff DateAttribute Logic Error

**Target**: `scipy.io.arff._arffread.DateAttribute._get_date_format`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_get_date_format` method in `DateAttribute` class contains a logic error at line 276 where `elif "yy":` is used instead of `elif "yy" in pattern:`. This causes the condition to always evaluate to True (since non-empty strings are truthy in Python), leading to incorrect code execution flow even though the bug is masked in most cases by subsequent overwrites.

## Property-Based Test

```python
from scipy.io.arff._arffread import DateAttribute
from hypothesis import given, strategies as st, assume


@given(st.text(alphabet='MdHms-/: ', min_size=1, max_size=50))
def test_date_format_without_year_shouldnt_set_year_unit(pattern_body):
    """
    Test that date patterns without year components don't incorrectly
    set datetime_unit to 'Y' during intermediate processing.

    This property would fail if we could observe intermediate state,
    but passes in practice because later components overwrite the bug.
    """
    assume('y' not in pattern_body.lower())
    assume(any(x in pattern_body for x in ['M', 'd', 'H', 'm', 's']))

    pattern_str = f"date {pattern_body}"

    try:
        date_fmt, datetime_unit = DateAttribute._get_date_format(pattern_str)
        assert datetime_unit != "Y", \
            f"Pattern {pattern_body} has no year but datetime_unit is 'Y'"
    except ValueError:
        pass
```

**Failing input**: The bug is in the code logic itself, not triggered by specific input.

## Reproducing the Bug

```python
from scipy.io.arff._arffread import DateAttribute

test_pattern = "date MM-dd"
result_fmt, result_unit = DateAttribute._get_date_format(test_pattern)

print(f"Pattern: {test_pattern}")
print(f"Expected: datetime_unit should never be 'Y' during processing")
print(f"Actual: Line 276 'elif \"yy\":' always evaluates to True")
print(f"        causing datetime_unit='Y' to be set incorrectly")
print(f"        before being overwritten by later 'dd' check")

print(f"\nProof: bool('yy') = {bool('yy')} (always True!)")
```

## Why This Is A Bug

Line 276 of `_arffread.py` contains:

```python
elif "yy":
```

This condition is always `True` because the string `"yy"` is a non-empty string, which is truthy in Python. The correct code should be:

```python
elif "yy" in pattern:
```

**Execution flow for pattern "date MM-dd":**
1. Line 273: `"yyyy" in pattern` → False, skip if-branch
2. Line 276: `"yy"` → **Always True** (BUG!)
3. Line 277: `pattern.replace("yy", "%y")` → No change (harmless)
4. Line 278: `datetime_unit = "Y"` → **WRONG** (pattern has no year!)
5. Line 281: `datetime_unit = "M"` (overwrites incorrect "Y")
6. Line 284: `datetime_unit = "D"` (final correct value)

While the bug is masked by later overwrites in most cases, it represents:
- **Incorrect logic** that makes code confusing
- **Fragile design** that relies on overwrites to work correctly
- **Maintenance hazard** if code is refactored

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