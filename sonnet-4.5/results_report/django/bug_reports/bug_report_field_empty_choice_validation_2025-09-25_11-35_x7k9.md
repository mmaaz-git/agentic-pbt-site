# Bug Report: Field Validation Rejects Empty Values in Choices

**Target**: `django.db.models.fields.Field.validate`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Django field validation incorrectly rejects empty values (empty string, None) even when they are explicitly included as valid choices. The blank/null validation runs after choice validation but the choice validation is skipped for empty values, causing valid empty choices to be rejected.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.db.models.fields import CharField
from django.core.exceptions import ValidationError
import pytest


@given(
    st.lists(st.tuples(st.text(), st.text()), min_size=1),
    st.text()
)
def test_charfield_validates_choices(choices, value):
    """
    Property: CharField should accept values in choices, reject others.
    """
    field = CharField(choices=choices)
    choice_values = {choice[0] for choice in choices}

    if value in choice_values:
        field.validate(value, None)
    else:
        with pytest.raises(ValidationError):
            field.validate(value, None)
```

**Failing input**: `choices=[('', '')], value=''`

## Reproducing the Bug

```python
import django
from django.conf import settings
settings.configure(USE_I18N=True, USE_TZ=False)
django.setup()

from django.db.models.fields import CharField, IntegerField
from django.core.exceptions import ValidationError

field = CharField(choices=[('', 'Empty choice'), ('a', 'A choice')], blank=False)
field.validate('', None)

field2 = IntegerField(choices=[(None, 'None choice'), (1, 'One')], null=False)
field2.validate(None, None)
```

## Why This Is A Bug

The current validation logic in `Field.validate()`:

```python
if self.choices is not None and value not in self.empty_values:
    # Check if value is in choices
    ...

if not self.blank and value in self.empty_values:
    raise ValidationError(...)
```

This creates a logic flaw:
1. When `value = ''` (or `None`), it's in `self.empty_values`
2. Choice validation is skipped because `value not in self.empty_values` is False
3. Then blank/null validation runs and rejects the value
4. **Result**: Empty string/None is rejected even though it's a valid choice

**Expected behavior**: If a value is explicitly included in choices, it should be accepted regardless of blank/null settings. The choices constraint is more specific than the blank/null constraint.

## Fix

```diff
--- a/django/db/models/fields/__init__.py
+++ b/django/db/models/fields/__init__.py
@@ -808,7 +808,7 @@ class Field(RegisterLookupMixin):
         if not self.editable:
             # Skip validation for non-editable fields.
             return

-        if self.choices is not None and value not in self.empty_values:
+        if self.choices is not None:
             for option_key, option_value in self.choices:
                 if isinstance(option_value, (list, tuple)):
                     # This is an optgroup, so look inside the group for
@@ -817,6 +817,8 @@ class Field(RegisterLookupMixin):
                         if value == optgroup_key:
                             return
                 elif value == option_key:
                     return
+            # Value not found in choices
             raise exceptions.ValidationError(
                 self.error_messages["invalid_choice"],
                 code="invalid_choice",
@@ -824,6 +826,7 @@ class Field(RegisterLookupMixin):
             )

+        # Only check blank/null if value is not in choices
         if value is None and not self.null:
             raise exceptions.ValidationError(self.error_messages["null"], code="null")
```