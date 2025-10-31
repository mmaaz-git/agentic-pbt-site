# Bug Report: CreateModel.reduce() RenameField Case-Sensitive Comparison

**Target**: `django.db.migrations.operations.models.CreateModel.reduce()` (RenameField case)
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When `CreateModel.reduce()` processes a `RenameField` operation, it fails to properly rename the field in `unique_together` and `index_together` constraints when the field name in the constraint differs in case from the `old_name` parameter. The bug occurs because field names are compared case-sensitively.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
from django.db import models
from django.db.migrations.operations import CreateModel, RenameField

field_name_strategy = st.text(min_size=1, max_size=30, alphabet=st.characters(
    whitelist_categories=('Lu', 'Ll'), min_codepoint=65, max_codepoint=122
))

@given(
    old_field=field_name_strategy,
    new_field=field_name_strategy,
    constraint_field_variant=field_name_strategy
)
@settings(max_examples=200)
def test_rename_field_case_insensitive_in_constraints(old_field, new_field, constraint_field_variant):
    assume(old_field.lower() != new_field.lower())
    assume(constraint_field_variant.lower() == old_field.lower())
    assume(constraint_field_variant != old_field)

    create_op = CreateModel(
        name='MyModel',
        fields=[
            ('id', models.AutoField(primary_key=True)),
            ('other', models.CharField(max_length=100)),
            (old_field, models.CharField(max_length=100))
        ],
        options={'unique_together': {('other', constraint_field_variant)}}
    )

    rename_op = RenameField(
        model_name='MyModel',
        old_name=old_field,
        new_name=new_field
    )

    result = create_op.reduce(rename_op, app_label='test_app')
    unique_together = result[0].options.get('unique_together')

    for tup in unique_together:
        for f in tup:
            assert f.lower() != old_field.lower() or f.lower() == new_field.lower(), \
                f"Field not renamed: {f} should be {new_field}"
```

**Failing input**: `old_field='myField'`, `constraint_field_variant='MyField'`, `new_field='renamedField'`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.db import models
from django.db.migrations.operations import CreateModel, RenameField

create_op = CreateModel(
    name='MyModel',
    fields=[
        ('id', models.AutoField(primary_key=True)),
        ('other', models.CharField(max_length=100)),
        ('myField', models.CharField(max_length=100))
    ],
    options={'unique_together': {('other', 'MyField')}}
)

rename_op = RenameField(
    model_name='MyModel',
    old_name='myField',
    new_name='renamedField'
)

result = create_op.reduce(rename_op, app_label='test_app')

print(result[0].options.get('unique_together'))
```

**Expected output**: `{('other', 'renamedField')}`
**Actual output**: `{('other', 'MyField')}` (field NOT renamed - BUG!)

## Why This Is A Bug

When renaming a field, Django migrations should update all references to that field in constraints like `unique_together` and `index_together`. The `CreateModel.reduce()` method attempts to do this, but uses case-sensitive comparison:

```python
operation.new_name if f == operation.old_name else f
```

Since Django field names are case-insensitive (they get lowercased in the database), a field defined as `myField` might be referenced as `MyField` in `unique_together`. The case-sensitive comparison fails to match them, leaving the old field name in the constraint, which references a non-existent field after the rename.

This can cause:
1. Invalid database migrations (referencing non-existent fields)
2. Migration optimizer producing incorrect optimized migrations
3. Database constraint errors

## Fix

The fix is to perform case-insensitive comparison in the field renaming logic. In `/django/db/migrations/operations/models.py` at lines 329-332, change:

```diff
                         tuple(
-                            operation.new_name if f == operation.old_name else f
+                            operation.new_name if f.lower() == operation.old_name_lower else f
                             for f in fields
                         )
```

Also update the `order_with_respect_to` comparison at line 336:

```diff
                     order_with_respect_to = options.get("order_with_respect_to")
-                    if order_with_respect_to == operation.old_name:
+                    if order_with_respect_to and order_with_respect_to.lower() == operation.old_name_lower:
                         options["order_with_respect_to"] = operation.new_name
```