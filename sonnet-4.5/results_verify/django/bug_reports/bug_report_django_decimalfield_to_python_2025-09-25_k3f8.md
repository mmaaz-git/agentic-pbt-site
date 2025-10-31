# Bug Report: DecimalField.to_python() Inconsistent Float vs String Conversion

**Target**: `django.db.models.fields.DecimalField.to_python`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

DecimalField.to_python() produces different results when converting a float value versus converting the string representation of that same float value, violating the expected consistency of type conversion.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from django.db.models.fields import DecimalField

@given(st.floats(allow_nan=False, allow_infinity=False), st.integers(min_value=1, max_value=20), st.integers(min_value=0, max_value=10))
def test_decimalfield_float_vs_string_conversion(float_value, max_digits, decimal_places):
    assume(decimal_places <= max_digits)

    field = DecimalField(max_digits=max_digits, decimal_places=decimal_places)

    try:
        result_from_float = field.to_python(float_value)
        result_from_string = field.to_python(str(float_value))

        assert result_from_float == result_from_string
    except (ValidationError, decimal.InvalidOperation):
        pass
```

**Failing input**: `float_value=11.0, max_digits=1, decimal_places=0`

## Reproducing the Bug

```python
from django.db.models.fields import DecimalField

field = DecimalField(max_digits=1, decimal_places=0)

result_from_float = field.to_python(11.0)
result_from_string = field.to_python("11.0")

print(result_from_float)
print(result_from_string)
print(result_from_float == result_from_string)
```

**Output**:
```
1E+1
11.0
False
```

## Why This Is A Bug

The `to_python()` method is documented to "Convert the input value into the expected Python data type". For equivalent input values (a float and its string representation), it should produce the same result.

The inconsistency arises because:
1. Float inputs use `self.context.create_decimal_from_float(value)` where `context` has precision set to `max_digits`
2. String inputs use `decimal.Decimal(value)` which doesn't apply the precision limit
3. This causes 11.0 → Decimal('1E+1') via float path but "11.0" → Decimal('11.0') via string path

This violates the principle of consistent type conversion and could cause subtle bugs when users pass semantically equivalent values in different formats.

## Fix

The fix should ensure consistent conversion regardless of input format. The most straightforward approach is to not use the context for float conversion, or apply the same precision constraints to both paths:

```diff
--- a/django/db/models/fields/__init__.py
+++ b/django/db/models/fields/__init__.py
@@ -1812,7 +1812,7 @@ class DecimalField(Field):
             return value
         try:
             if isinstance(value, float):
-                decimal_value = self.context.create_decimal_from_float(value)
+                decimal_value = decimal.Decimal(str(value))
             else:
                 decimal_value = decimal.Decimal(value)
         except (decimal.InvalidOperation, TypeError, ValueError):
```

This ensures both float and string inputs are converted consistently by first converting floats to strings before creating Decimal objects. The precision validation is then handled consistently by the DecimalValidator during the validation phase.