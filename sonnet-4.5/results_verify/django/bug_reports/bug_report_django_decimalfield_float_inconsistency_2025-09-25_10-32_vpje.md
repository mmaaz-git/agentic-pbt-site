# Bug Report: DecimalField Float vs String Inconsistency

**Target**: `django.db.models.fields.DecimalField.to_python`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

DecimalField accepts float values that would be rejected if passed as strings, due to precision-based rounding that occurs before validation for floats but not for strings.

## Property-Based Test

```python
from decimal import Decimal
from hypothesis import given, strategies as st
from django.db.models.fields import DecimalField
from django.core.exceptions import ValidationError


field = DecimalField(max_digits=10, decimal_places=2)


@given(st.floats(allow_nan=False, allow_infinity=False, min_value=1e7, max_value=1e8))
def test_decimal_float_vs_string_consistency(f):
    str_val = str(f)

    try:
        result_from_float = field.to_python(f)
        field.run_validators(result_from_float)
        float_accepted = True
    except ValidationError:
        float_accepted = False

    try:
        result_from_string = field.to_python(str_val)
        field.run_validators(result_from_string)
        string_accepted = True
    except ValidationError:
        string_accepted = False

    assert float_accepted == string_accepted
```

**Failing input**: `10000000.375`

## Reproducing the Bug

```python
import django
from django.conf import settings

if not settings.configured:
    settings.configure(
        DATABASES={'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': ':memory:'}},
        INSTALLED_APPS=['django.contrib.contenttypes'],
        USE_TZ=True,
    )
    django.setup()

from django.db.models.fields import DecimalField
from django.core.exceptions import ValidationError

field = DecimalField(max_digits=10, decimal_places=2)

value = 10000000.375

result_float = field.to_python(value)
field.run_validators(result_float)
print(f"Float {value} accepted: {result_float}")

try:
    result_string = field.to_python(str(value))
    field.run_validators(result_string)
    print(f"String '{value}' accepted: {result_string}")
except ValidationError as e:
    print(f"String '{value}' rejected: {e}")
```

Output:
```
Float 10000000.375 accepted: 10000000.38
String '10000000.375' rejected: ['Ensure that there are no more than 10 digits in total.']
```

## Why This Is A Bug

When a DecimalField with `max_digits=10, decimal_places=2` receives a value:
- As a **float** (e.g., `10000000.375`): The value is converted using `context.create_decimal_from_float()`, which rounds to 10 significant digits (the precision), producing `Decimal('10000000.38')`. This passes validation.
- As a **string** (e.g., `'10000000.375'`): The value is converted using `Decimal(value)`, which preserves all digits, producing `Decimal('10000000.375')` with 11 total digits. This fails validation.

This inconsistency means semantically identical values are handled differently based on their input type. Users would reasonably expect that `field.to_python(x)` produces the same result (or both succeed/fail) for numerically equivalent inputs, regardless of whether they're passed as floats, strings, or Decimals.

## Fix

The fix should ensure consistent behavior across input types. The field should either:
1. Round all inputs to the configured constraints before validation, or
2. Reject all inputs that exceed constraints, regardless of type

Option 1 (make behavior consistent by rounding all types):

```diff
--- a/django/db/models/fields/__init__.py
+++ b/django/db/models/fields/__init__.py
@@ -1803,10 +1803,13 @@ class DecimalField(Field):
     def to_python(self, value):
         if value is None:
             return value
         try:
             if isinstance(value, float):
                 decimal_value = self.context.create_decimal_from_float(value)
             else:
                 decimal_value = decimal.Decimal(value)
+                # Apply same precision rounding for consistency
+                decimal_value = self.context.create_decimal(decimal_value)
         except (decimal.InvalidOperation, TypeError, ValueError):
             raise exceptions.ValidationError(
                 self.error_messages["invalid"],
```

However, this may not be the desired behavior. Option 2 (reject floats with too much precision) might be more appropriate, but would require checking the float's precision before conversion.