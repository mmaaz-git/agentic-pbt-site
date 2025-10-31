# Bug Report: django.db.models.fields.Field validate() Incorrectly Rejects None When null=True, blank=False

**Target**: `django.db.models.fields.Field.validate()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `Field.validate()` method incorrectly raises a "blank" validation error for `None` values when a field has `null=True` and `blank=False`. This violates the expected behavior where `null=True` should allow None values regardless of the `blank` setting.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.db.models import fields
from django.core import exceptions
import pytest

@given(null=st.booleans())
def test_field_null_validation(null):
    field = fields.IntegerField(null=null)

    if null:
        field.validate(None, None)
    else:
        with pytest.raises(exceptions.ValidationError):
            field.validate(None, None)
```

**Failing input**: `null=True`

## Reproducing the Bug

```python
import django
from django.conf import settings

if not settings.configured:
    settings.configure(
        DEBUG=True,
        DATABASES={'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': ':memory:'}},
        INSTALLED_APPS=[],
    )
    django.setup()

from django.db.models import fields
from django.core import exceptions

field = fields.IntegerField(null=True, blank=False)

try:
    field.validate(None, None)
    print("Validation passed")
except exceptions.ValidationError as e:
    print(f"Validation failed: {e}")
    print(f"Error code: {e.code}")
```

Output:
```
Validation failed: ['This field cannot be blank.']
Error code: blank
```

## Why This Is A Bug

In Django's ORM:
- `null=True` means the database column allows NULL values
- `blank=False` means the field requires a value in forms/validation

When `null=True`, the field explicitly allows `None` as a valid value at the database level. The `validate()` method should respect this and not raise a "blank" error for `None` values.

The current implementation at `/django/db/models/fields/__init__.py:823-827` has this logic:

```python
if value is None and not self.null:
    raise exceptions.ValidationError(self.error_messages["null"], code="null")

if not self.blank and value in self.empty_values:
    raise exceptions.ValidationError(self.error_messages["blank"], code="blank")
```

The problem: `None` is in `empty_values`, so even when `null=True` (passing the first check), the `None` value still triggers the blank check (failing the second check).

## Fix

The blank check should exclude `None` values when `null=True`:

```diff
--- a/django/db/models/fields/__init__.py
+++ b/django/db/models/fields/__init__.py
@@ -823,7 +823,7 @@ class Field(RegisterLookupMixin):
         if value is None and not self.null:
             raise exceptions.ValidationError(self.error_messages["null"], code="null")

-        if not self.blank and value in self.empty_values:
+        if not self.blank and value in self.empty_values and value is not None:
             raise exceptions.ValidationError(self.error_messages["blank"], code="blank")

     def clean(self, value, model_instance):
```

This ensures that `None` values are only validated by the `null` check, not the `blank` check.