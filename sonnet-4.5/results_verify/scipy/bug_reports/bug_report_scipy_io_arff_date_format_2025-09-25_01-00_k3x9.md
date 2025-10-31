# Bug Report: scipy.io.arff DateAttribute Invalid Format Validation

**Target**: `scipy.io.arff._arffread.DateAttribute._get_date_format`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `DateAttribute._get_date_format` method incorrectly accepts invalid date format strings that contain no recognized date/time components, due to a logic error on line 276 where `elif "yy":` should be `elif "yy" in pattern:`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from scipy.io.arff._arffread import DateAttribute

@given(st.text(alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=1, max_size=20))
def test_date_format_validation(random_text):
    # Assume the text doesn't accidentally contain valid date components
    from hypothesis import assume
    valid_components = ["yyyy", "yy", "MM", "dd", "HH", "mm", "ss"]
    assume(not any(comp in random_text for comp in valid_components))
    assume("z" not in random_text.lower())  # timezone check

    date_str = f"date {random_text}"

    try:
        format_pattern, unit = DateAttribute._get_date_format(date_str)
        # If we get here without error, it's a bug
        assert False, f"Should have raised ValueError for invalid format '{random_text}', but got format={format_pattern}, unit={unit}"
    except ValueError as e:
        # This is expected
        assert "Invalid or unsupported date format" in str(e)
```

**Failing input**: `random_text='abc'`

## Reproducing the Bug

```python
from scipy.io.arff._arffread import DateAttribute

invalid_format = "date abc"
format_pattern, unit = DateAttribute._get_date_format(invalid_format)

print(f"Format: {format_pattern}")
print(f"Unit: {unit}")
```

Output:
```
Format: abc
Unit: Y
```

Expected: Should raise `ValueError("Invalid or unsupported date format")`

## Why This Is A Bug

The code on line 276 has `elif "yy":` which evaluates the truthiness of the string literal `"yy"` (always True), instead of checking `elif "yy" in pattern:`. This causes `datetime_unit` to be unconditionally set to `"Y"` for any pattern that doesn't contain `"yyyy"`, even if the pattern contains no date/time components at all.

As a result, invalid date format strings like `"date abc"` or `"date foobar"` are accepted when they should raise a ValueError on line 298-299.

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