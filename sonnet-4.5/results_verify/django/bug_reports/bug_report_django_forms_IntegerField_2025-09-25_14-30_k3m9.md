# Bug Report: django.forms.IntegerField Rejects Boolean Values

**Target**: `django.forms.IntegerField`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

IntegerField rejects boolean values (True/False) even though Python's `int()` function accepts booleans and the field's docstring states it validates "that int() can be called on the input."

## Property-Based Test

```python
from hypothesis import given, strategies as st
import django.forms as forms


@given(st.booleans())
def test_integerfield_should_accept_booleans(b):
    """
    Property: IntegerField should accept boolean values since int() can be called on booleans.

    The IntegerField docstring states: "Validate that int() can be called on the input."
    Since int(True) == 1 and int(False) == 0 are valid Python operations,
    IntegerField should accept boolean values.
    """
    field = forms.IntegerField()

    try:
        result = field.clean(b)
        expected = int(b)
        assert result == expected, f"Expected {expected}, got {result}"
    except forms.ValidationError:
        assert False, f"IntegerField should accept boolean {b} since int({b}) == {int(b)}"
```

**Failing input**: `False` (also fails for `True`)

## Reproducing the Bug

```python
import django
from django.conf import settings

if not settings.configured:
    settings.configure(DEBUG=True, SECRET_KEY='test', USE_I18N=True)
    django.setup()

import django.forms as forms

field = forms.IntegerField()

print(f"int(True) = {int(True)}")
print(f"int(False) = {int(False)}")

try:
    result = field.clean(True)
    print(f"field.clean(True) = {result}")
except forms.ValidationError as e:
    print(f"field.clean(True) raised ValidationError: {e.messages}")

try:
    result = field.clean(False)
    print(f"field.clean(False) = {result}")
except forms.ValidationError as e:
    print(f"field.clean(False) raised ValidationError: {e.messages}")
```

Output:
```
int(True) = 1
int(False) = 0
field.clean(True) raised ValidationError: ['Enter a whole number.']
field.clean(False) raised ValidationError: ['Enter a whole number.']
```

## Why This Is A Bug

1. **Contract Violation**: The IntegerField.to_python() docstring explicitly states: "Validate that int() can be called on the input." In Python, `int(True) == 1` and `int(False) == 0` are valid operations.

2. **Inconsistent with Python semantics**: Booleans are a subclass of int in Python (`bool.__bases__ == (int,)`), so they should be accepted where integers are expected.

3. **Unexpected behavior**: Users who pass boolean values programmatically (e.g., in API validation or data processing) would reasonably expect them to be converted to 0 or 1.

4. **Root cause**: The implementation converts values to strings before parsing:
   ```python
   value = int(self.re_decimal.sub("", str(value)))
   ```
   This causes `True` → `"True"` → `ValueError` instead of `True` → `1`.

## Fix

```diff
--- a/django/forms/fields.py
+++ b/django/forms/fields.py
@@ -341,6 +341,10 @@ class IntegerField(Field):
         value = super().to_python(value)
         if value in self.empty_values:
             return None
+        # Handle booleans before string conversion since bool is a subclass of int
+        # and int(True) == 1, int(False) == 0 are valid
+        if isinstance(value, bool):
+            return int(value)
         if self.localize:
             value = formats.sanitize_separators(value)
         # Strip trailing decimal and zeros.
```

This fix checks if the value is a boolean before attempting string conversion, allowing booleans to be properly converted to their integer equivalents (True → 1, False → 0).