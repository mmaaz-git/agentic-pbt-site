# Bug Report: django.template.defaultfilters.get_digit ValueError on Negative Numbers

**Target**: `django.template.defaultfilters.get_digit`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `get_digit` filter crashes with `ValueError: invalid literal for int() with base 10: '-'` when given a negative number and a position that accesses the minus sign character.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.template import defaultfilters


@given(st.integers(min_value=-1000, max_value=-1), st.integers(min_value=1, max_value=10))
def test_get_digit_negative_numbers(value, position):
    result = defaultfilters.get_digit(value, position)
    value_str = str(value)
    try:
        expected = int(value_str[-position])
    except (IndexError, ValueError):
        expected = 0
    assert result == expected or result == value
```

**Failing input**: `get_digit(-123, 4)` or `get_digit(-1, 2)`

## Reproducing the Bug

```python
from django.template import defaultfilters

result = defaultfilters.get_digit(-123, 4)
```

This crashes with:
```
ValueError: invalid literal for int() with base 10: '-'
```

The bug occurs because `str(-123)` is `"-123"` (4 characters), and `str(-123)[-4]` returns `"-"`. The code then tries `int("-")` which raises `ValueError`.

## Why This Is A Bug

According to the docstring, `get_digit` should "Return the original value for invalid input" and "output is always an integer" (or the original value). The function has a try-except block for `IndexError` to return 0 when the position is out of range, but it doesn't handle the case where the position accesses the minus sign.

This violates the documented behavior of failing silently and always returning an integer or the original value.

## Fix

```diff
--- a/django/template/defaultfilters.py
+++ b/django/template/defaultfilters.py
@@ -765,7 +765,11 @@ def get_digit(value, arg):
     if arg < 1:
         return value
     try:
-        return int(str(value)[-arg])
+        digit_char = str(value)[-arg]
+        if not digit_char.isdigit():
+            return 0
+        return int(digit_char)
     except IndexError:
         return 0
```

Alternatively, a more minimal fix that just catches the ValueError:

```diff
--- a/django/template/defaultfilters.py
+++ b/django/template/defaultfilters.py
@@ -765,7 +765,7 @@ def get_digit(value, arg):
     if arg < 1:
         return value
     try:
         return int(str(value)[-arg])
-    except IndexError:
+    except (IndexError, ValueError):
         return 0
```