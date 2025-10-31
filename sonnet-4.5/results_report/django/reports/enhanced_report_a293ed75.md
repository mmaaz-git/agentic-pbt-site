# Bug Report: django.db.migrations.operations FieldOperation.references_field Case Sensitivity Inconsistency

**Target**: `django.db.migrations.operations.fields.FieldOperation.references_field`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `FieldOperation.references_field()` method uses case-sensitive field name comparison while all other similar methods in the same class use case-insensitive comparison, causing inconsistent behavior that breaks migration operation optimization.

## Property-Based Test

```python
import django
from django.conf import settings

if not settings.configured:
    settings.configure(INSTALLED_APPS=[], SECRET_KEY='test', USE_TZ=True)
    django.setup()

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

    assert is_same == references, f"is_same_field_operation={is_same} but references_field={references} for base_name='{base_name}', variant_name='{variant_name}'"

if __name__ == "__main__":
    # Run the test
    test_is_same_vs_references_consistency()
```

<details>

<summary>
**Failing input**: `('aaa', 'aaA')`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 34, in <module>
    test_is_same_vs_references_consistency()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 21, in test_is_same_vs_references_consistency
    def test_is_same_vs_references_consistency(names):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 30, in test_is_same_vs_references_consistency
    assert is_same == references, f"is_same_field_operation={is_same} but references_field={references} for base_name='{base_name}', variant_name='{variant_name}'"
           ^^^^^^^^^^^^^^^^^^^^^
AssertionError: is_same_field_operation=True but references_field=False for base_name='aaa', variant_name='aaA'
Falsifying example: test_is_same_vs_references_consistency(
    names=('aaa', 'aaA'),  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import django
from django.conf import settings

if not settings.configured:
    settings.configure(INSTALLED_APPS=[], SECRET_KEY='test', USE_TZ=True)
    django.setup()

from django.db import models
from django.db.migrations.operations import AddField

# Create a CharField
field = models.CharField(max_length=100)

# Create an AddField operation with field name "firstName"
op = AddField(model_name="User", name="firstName", field=field)

# Create another AddField operation with field name in uppercase "FIRSTNAME"
op2 = AddField(model_name="User", name="FIRSTNAME", field=field)

# Test is_same_field_operation - expects True due to case-insensitive comparison
print(f"op.is_same_field_operation(op2): {op.is_same_field_operation(op2)}")

# Test references_field with uppercase name - expects True but returns False
print(f"op.references_field('User', 'FIRSTNAME', 'app'): {op.references_field('User', 'FIRSTNAME', 'app')}")

# The inconsistency: is_same_field_operation says they're the same field,
# but references_field says the first operation doesn't reference the second
print("\nInconsistency detected:")
print(f"  is_same_field_operation returns: {op.is_same_field_operation(op2)}")
print(f"  references_field returns: {op.references_field('User', 'FIRSTNAME', 'app')}")
print(f"  Expected: Both should return True")
```

<details>

<summary>
Inconsistency detected between is_same_field_operation and references_field
</summary>
```
op.is_same_field_operation(op2): True
op.references_field('User', 'FIRSTNAME', 'app'): False

Inconsistency detected:
  is_same_field_operation returns: True
  references_field returns: False
  Expected: Both should return True
```
</details>

## Why This Is A Bug

Django's migration system consistently treats field names as case-insensitive identifiers throughout the framework. The `FieldOperation` class itself demonstrates this pattern:

1. **Cached properties for lowercase names** (lines 19-20): The class maintains `name_lower` property specifically for case-insensitive comparisons
2. **is_same_field_operation()** (line 28): Uses `self.name_lower == operation.name_lower` for case-insensitive comparison
3. **references_model()** (line 32): Uses `name.lower() == self.model_name_lower` for case-insensitive comparison
4. **RenameField.references_field()** (line 344): Overrides parent to use `name.lower() == self.old_name_lower`

However, `FieldOperation.references_field()` at line 49 breaks this pattern with `if name == self.name:` - a case-sensitive comparison. This inconsistency violates the established contract and can cause migration optimizations to fail when dealing with field names that differ only in case. The fact that `RenameField` had to override this method to fix the case sensitivity issue indicates this is a known problem that was only partially addressed.

## Relevant Context

The Django migration system is designed to be database-agnostic and handle various database backends that may have different case sensitivity rules. Many databases treat identifiers as case-insensitive (e.g., MySQL, PostgreSQL with unquoted identifiers), so Django's migration system adopts case-insensitive comparisons as the standard approach.

This bug can manifest when:
- Working with legacy databases where field naming conventions have changed over time
- Integrating Django with external systems that use different casing conventions
- Performing migration squashing or optimization where field operations need to be correctly identified

The bug is located in the Django source at:
`django/db/migrations/operations/fields.py`, line 49

Django documentation on migrations: https://docs.djangoproject.com/en/stable/topics/migrations/

## Proposed Fix

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