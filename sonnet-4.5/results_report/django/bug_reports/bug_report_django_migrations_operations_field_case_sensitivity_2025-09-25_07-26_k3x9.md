# Bug Report: django.db.migrations.operations FieldOperation references_field Case Sensitivity Inconsistency

**Target**: `django.db.migrations.operations.fields.FieldOperation.references_field`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `FieldOperation.references_field()` method uses case-sensitive field name comparison, while other methods in the same class (`is_same_field_operation`, `is_same_model_operation`, `references_model`) use case-insensitive comparison. This inconsistency can break operation optimization when field names differ only in case.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from django.db import models
from django.db.migrations.operations import AddField
import string

@st.composite
def field_name_case_variants(draw):
    base_name = draw(st.text(alphabet=string.ascii_lowercase, min_size=3, max_size=10))
    variant = ''.join(c.upper() if draw(st.booleans()) else c for c in base_name)
    assume(base_name != variant)
    return base_name, variant

@given(field_name_case_variants())
def test_is_same_vs_references_consistency(names):
    base_name, variant_name = names
    field = models.CharField(max_length=100)
    op1 = AddField(model_name="TestModel", name=base_name, field=field)
    op2 = AddField(model_name="TestModel", name=variant_name, field=field)

    is_same = op1.is_same_field_operation(op2)
    references = op1.references_field("TestModel", variant_name, "testapp")

    assert is_same == references, f"is_same_field_operation={is_same} but references_field={references}"
```

**Failing input**: `base_name='abc'`, `variant_name='Abc'`

## Reproducing the Bug

```python
import django
from django.conf import settings

if not settings.configured:
    settings.configure(INSTALLED_APPS=[], SECRET_KEY='test', USE_TZ=True)
    django.setup()

from django.db import models
from django.db.migrations.operations import AddField

field = models.CharField(max_length=100)
op = AddField(model_name="User", name="firstName", field=field)

op2 = AddField(model_name="User", name="FIRSTNAME", field=field)
print(f"is_same_field_operation: {op.is_same_field_operation(op2)}")

print(f"references_field('User', 'FIRSTNAME', 'app'): {op.references_field('User', 'FIRSTNAME', 'app')}")
```

## Why This Is A Bug

Django's migration system consistently uses case-insensitive field name comparisons throughout:
- `is_same_field_operation()` compares `self.name_lower == operation.name_lower`
- `references_model()` compares `name.lower() == self.model_name_lower`
- `RenameField.references_field()` compares `name.lower() == self.old_name_lower`

However, `FieldOperation.references_field()` at line 49 uses:
```python
if name == self.name:
```

This case-sensitive comparison is inconsistent with the rest of the codebase and Django's general treatment of field names as case-insensitive identifiers. This can cause operation optimization failures when field names differ only in case.

## Fix

```diff
--- a/django/db/migrations/operations/fields.py
+++ b/django/db/migrations/operations/fields.py
@@ -46,7 +46,7 @@ class FieldOperation(Operation):
         model_name_lower = model_name.lower()
         # Check if this operation locally references the field.
         if model_name_lower == self.model_name_lower:
-            if name == self.name:
+            if name.lower() == self.name_lower:
                 return True
             elif (
                 self.field
```