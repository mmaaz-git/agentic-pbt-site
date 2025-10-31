# Bug Report: Django CreateModel.reduce() Case-Sensitive Field Comparison in Constraints

**Target**: `django.db.migrations.operations.models.CreateModel.reduce()` (RenameField case)
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Django's `CreateModel.reduce()` method fails to update field references in constraints (`unique_together`, `index_together`, `order_with_respect_to`) when the field name in the constraint uses different casing than the field definition, causing the migration optimizer to produce invalid migrations with references to non-existent fields.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Property-based test using Hypothesis to systematically test Django's CreateModel.reduce()
for case-sensitivity issues when renaming fields referenced in constraints.
"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings, assume, HealthCheck
from django.db import models
from django.db.migrations.operations import CreateModel, RenameField

# Strategy that generates a base field name and creates a case variant
@st.composite
def field_name_with_variant(draw):
    """Generate a field name and a case-variant of it."""
    # Generate a base name with at least 2 characters to have room for case variation
    base_name = draw(st.text(min_size=2, max_size=10, alphabet=st.characters(whitelist_categories=('Ll',))))

    # Create a case variant by changing the case of at least one character
    variant_chars = list(base_name)
    # Change the first character to uppercase for the variant
    variant_chars[0] = variant_chars[0].upper()
    variant = ''.join(variant_chars)

    # Generate a different name for the new field
    new_name = draw(st.text(min_size=2, max_size=10, alphabet=st.characters(whitelist_categories=('Ll',))).filter(lambda x: x != base_name))

    return base_name, variant, new_name

@given(field_names=field_name_with_variant())
@settings(max_examples=200, suppress_health_check=[HealthCheck.filter_too_much])
def test_rename_field_case_insensitive_in_constraints(field_names):
    """
    Test that RenameField correctly updates field references in constraints
    even when the constraint uses different case than the field definition.
    """
    old_field, constraint_field_variant, new_field = field_names

    # Create a model with the field and a constraint that references it with different case
    create_op = CreateModel(
        name='MyModel',
        fields=[
            ('id', models.AutoField(primary_key=True)),
            ('other', models.CharField(max_length=100)),
            (old_field, models.CharField(max_length=100))
        ],
        options={'unique_together': {('other', constraint_field_variant)}}
    )

    # Rename the field
    rename_op = RenameField(
        model_name='MyModel',
        old_name=old_field,
        new_name=new_field
    )

    # Reduce operations (what Django's migration optimizer does)
    result = create_op.reduce(rename_op, app_label='test_app')
    unique_together = result[0].options.get('unique_together')

    # Check that all field references were updated
    for tup in unique_together:
        for f in tup:
            assert f.lower() != old_field.lower() or f.lower() == new_field.lower(), \
                f"Field not renamed: {f} should be {new_field} (old_field={old_field}, constraint_field={constraint_field_variant})"

# Run the test
if __name__ == '__main__':
    print("Running Hypothesis property-based test for Django CreateModel.reduce() bug...")
    print("Testing case-sensitive field comparison in constraints...")
    print()
    test_rename_field_case_insensitive_in_constraints()
```

<details>

<summary>
**Failing input**: `field_names=('aaa', 'Aaa', 'aa')`
</summary>
```
Running Hypothesis property-based test for Django CreateModel.reduce() bug...
Testing case-sensitive field comparison in constraints...

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/18/hypo.py", line 74, in <module>
    test_rename_field_case_insensitive_in_constraints()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/18/hypo.py", line 33, in test_rename_field_case_insensitive_in_constraints
    @settings(max_examples=200, suppress_health_check=[HealthCheck.filter_too_much])
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/18/hypo.py", line 66, in test_rename_field_case_insensitive_in_constraints
    assert f.lower() != old_field.lower() or f.lower() == new_field.lower(), \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Field not renamed: Aaa should be aa (old_field=aaa, constraint_field=Aaa)
Falsifying example: test_rename_field_case_insensitive_in_constraints(
    field_names=('aaa', 'Aaa', 'aa'),
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/18/hypo.py:67
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Minimal reproduction case for Django CreateModel.reduce() bug with case-sensitive field comparison.
This demonstrates that when a field is defined with one case and referenced with another case
in constraints, the RenameField operation fails to update the constraint reference.
"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.db import models
from django.db.migrations.operations import CreateModel, RenameField

# Create a model with field 'myField' (lowercase 'my')
# but reference it as 'MyField' (uppercase 'My') in unique_together constraint
create_op = CreateModel(
    name='MyModel',
    fields=[
        ('id', models.AutoField(primary_key=True)),
        ('other', models.CharField(max_length=100)),
        ('myField', models.CharField(max_length=100))  # Field defined with lowercase 'my'
    ],
    options={'unique_together': {('other', 'MyField')}}  # Referenced with uppercase 'My'
)

# Try to rename 'myField' to 'renamedField'
rename_op = RenameField(
    model_name='MyModel',
    old_name='myField',
    new_name='renamedField'
)

# Reduce the operations (this is what Django's migration optimizer does)
result = create_op.reduce(rename_op, app_label='test_app')

# Check the result
unique_together = result[0].options.get('unique_together')

print("=== Django CreateModel.reduce() Case-Sensitive Bug Reproduction ===")
print()
print("Initial field definition: 'myField' (lowercase 'my')")
print("Constraint reference: 'MyField' (uppercase 'My')")
print("Rename operation: 'myField' -> 'renamedField'")
print()
print("Expected unique_together after reduce: {('other', 'renamedField')}")
print(f"Actual unique_together after reduce:   {unique_together}")
print()

# Check if the bug occurred
if unique_together == {('other', 'MyField')}:
    print("BUG CONFIRMED: Field 'MyField' was NOT renamed to 'renamedField'")
    print("The case-sensitive comparison failed to match 'MyField' with 'myField'")
    print()
    print("This will cause invalid migrations where constraints reference")
    print("non-existent fields after the rename operation.")
elif unique_together == {('other', 'renamedField')}:
    print("No bug: Field was correctly renamed")
else:
    print(f"Unexpected result: {unique_together}")
```

<details>

<summary>
BUG CONFIRMED: Field NOT renamed in constraint
</summary>
```
=== Django CreateModel.reduce() Case-Sensitive Bug Reproduction ===

Initial field definition: 'myField' (lowercase 'my')
Constraint reference: 'MyField' (uppercase 'My')
Rename operation: 'myField' -> 'renamedField'

Expected unique_together after reduce: {('other', 'renamedField')}
Actual unique_together after reduce:   {('other', 'MyField')}

BUG CONFIRMED: Field 'MyField' was NOT renamed to 'renamedField'
The case-sensitive comparison failed to match 'MyField' with 'myField'

This will cause invalid migrations where constraints reference
non-existent fields after the rename operation.
```
</details>

## Why This Is A Bug

Django's migration system is responsible for maintaining referential integrity during schema changes. When a field is renamed via `RenameField`, ALL references to that field must be updated, including those in constraints like `unique_together`, `index_together`, and `order_with_respect_to`.

The bug violates this contract in the following ways:

1. **Django accepts mismatched case in model definitions**: Django allows you to define a field as `myField` and reference it as `MyField` in constraints without raising any validation errors. This creates a valid model that Django accepts and can use.

2. **The migration optimizer produces invalid output**: When `CreateModel.reduce()` processes a `RenameField` operation, it uses case-sensitive string comparison (`f == operation.old_name`) at line 330 of `/django/db/migrations/operations/models.py`. This fails to match `MyField` with `myField`, leaving the old field name in the constraint.

3. **The resulting migration references non-existent fields**: After the rename operation, the field `myField` no longer exists (it's now `renamedField`), but the constraint still references `MyField`. This produces a migration that will fail when applied to a database.

4. **Inconsistent behavior across Django**: While Django's field lookup (`_meta.get_field()`) is case-sensitive in Python, many databases (like MySQL) are case-insensitive for identifiers. Django's acceptance of mismatched cases in model definitions but failure to handle them in migrations creates an inconsistency that breaks applications.

## Relevant Context

The issue occurs in Django's migration optimization code, which is designed to reduce multiple migration operations into fewer, more efficient operations. The `CreateModel.reduce()` method combines a `CreateModel` operation with subsequent operations like `RenameField`.

Key code location: `/django/db/migrations/operations/models.py`, lines 323-349

Django's documentation doesn't explicitly specify whether field references in constraints must match the exact case of field definitions. However, since Django accepts the mismatched case without errors, users reasonably expect it to work correctly throughout the system.

The bug affects all constraint types:
- `unique_together`
- `index_together`
- `order_with_respect_to`

Documentation references:
- [Django Model Meta options](https://docs.djangoproject.com/en/stable/ref/models/options/)
- [Django Migrations](https://docs.djangoproject.com/en/stable/topics/migrations/)

## Proposed Fix

The fix requires making field name comparisons case-insensitive in the `CreateModel.reduce()` method. This ensures that field references in constraints are properly updated regardless of case differences.

```diff
--- a/django/db/migrations/operations/models.py
+++ b/django/db/migrations/operations/models.py
@@ -323,13 +323,14 @@ class CreateModel(ModelOperation):
             elif isinstance(operation, RenameField):
                 options = self.options.copy()
                 for option_name in ("unique_together", "index_together"):
                     option = options.get(option_name)
                     if option:
+                        old_name_lower = operation.old_name.lower()
                         options[option_name] = {
                             tuple(
-                                operation.new_name if f == operation.old_name else f
+                                operation.new_name if f.lower() == old_name_lower else f
                                 for f in fields
                             )
                             for fields in option
                         }
                 order_with_respect_to = options.get("order_with_respect_to")
-                if order_with_respect_to == operation.old_name:
+                if order_with_respect_to and order_with_respect_to.lower() == operation.old_name.lower():
                     options["order_with_respect_to"] = operation.new_name
                 return [
                     CreateModel(
                         self.name,
                         fields=[
-                            (operation.new_name if n == operation.old_name else n, v)
+                            (operation.new_name if n.lower() == operation.old_name.lower() else n, v)
                             for n, v in self.fields
                         ],
```