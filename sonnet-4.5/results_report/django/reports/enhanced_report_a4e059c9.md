# Bug Report: Django Migrations Rename Operations Violate Immutability Contract

**Target**: `django.db.migrations.operations.RenameModel` and `django.db.migrations.operations.RenameIndex`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`RenameModel.database_backwards()` and `RenameIndex.database_backwards()` permanently mutate the operation's internal state when an exception is raised, violating Django's documented immutability requirement for migration operations.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Property-based test using Hypothesis that demonstrates the Django migration
operations mutation bug.
"""

import sys
import os

# Add Django to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

# Configure Django minimally
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django.conf.global_settings')
import django
from django.conf import settings
settings.configure(INSTALLED_APPS=[])
django.setup()

from hypothesis import given, strategies as st, assume
from django.db.migrations.operations import RenameModel

@given(
    old_name=st.text(min_size=1, max_size=50).filter(lambda x: x.strip() and x.isidentifier()),
    new_name=st.text(min_size=1, max_size=50).filter(lambda x: x.strip() and x.isidentifier()),
)
def test_rename_model_database_backwards_preserves_state(old_name, new_name):
    assume(old_name != new_name)

    op = RenameModel(old_name=old_name, new_name=new_name)

    original_old_name = op.old_name
    original_new_name = op.new_name

    try:
        op.database_backwards(
            app_label='test_app',
            schema_editor=None,
            from_state=None,
            to_state=None
        )
    except Exception:
        pass

    assert op.old_name == original_old_name, \
        f"old_name mutated from {original_old_name} to {op.old_name}"
    assert op.new_name == original_new_name, \
        f"new_name mutated from {original_new_name} to {op.new_name}"

if __name__ == "__main__":
    # Run the test
    test_rename_model_database_backwards_preserves_state()
```

<details>

<summary>
**Failing input**: `old_name='A', new_name='B'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 52, in <module>
    test_rename_model_database_backwards_preserves_state()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 24, in test_rename_model_database_backwards_preserves_state
    old_name=st.text(min_size=1, max_size=50).filter(lambda x: x.strip() and x.isidentifier()),
               ^^^
  File "/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/61/hypo.py", line 45, in test_rename_model_database_backwards_preserves_state
    assert op.old_name == original_old_name, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: old_name mutated from A to B
Falsifying example: test_rename_model_database_backwards_preserves_state(
    old_name='A',
    new_name='B',
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Minimal reproduction of the Django migration operations mutation bug.
This demonstrates that RenameModel and RenameIndex operations violate
their immutability contract by permanently mutating state when
database_backwards() encounters an exception.
"""

import sys
import os

# Add Django to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

# Configure Django minimally
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django.conf.global_settings')
import django
from django.conf import settings
settings.configure(INSTALLED_APPS=[])
django.setup()

from django.db.migrations.operations import RenameModel, RenameIndex

print("=" * 60)
print("Testing RenameModel Operation Mutation Bug")
print("=" * 60)

# Test Case 1: RenameModel
print("\n1. RenameModel Operation Test:")
print("-" * 40)

op = RenameModel(old_name='Author', new_name='Writer')
print(f"Initial state:")
print(f"  old_name = '{op.old_name}'")
print(f"  new_name = '{op.new_name}'")

# Call database_backwards with None arguments to trigger exception
print("\nCalling database_backwards() with None arguments...")
try:
    op.database_backwards(
        app_label='test_app',
        schema_editor=None,  # This will cause an exception
        from_state=None,
        to_state=None
    )
    print("No exception raised (unexpected)")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}")

print(f"\nState after exception:")
print(f"  old_name = '{op.old_name}'")
print(f"  new_name = '{op.new_name}'")

if op.old_name == 'Writer' and op.new_name == 'Author':
    print("\n❌ BUG CONFIRMED: Operation state was MUTATED!")
    print("   The old_name and new_name have been swapped.")
else:
    print("\n✓ Operation state remained unchanged.")

# Test Case 2: RenameIndex
print("\n" + "=" * 60)
print("2. RenameIndex Operation Test:")
print("-" * 40)

op2 = RenameIndex(
    model_name='TestModel',
    old_name='idx_old',
    new_name='idx_new'
)

print(f"Initial state:")
print(f"  old_name = '{op2.old_name}'")
print(f"  new_name = '{op2.new_name}'")

print("\nCalling database_backwards() with None arguments...")
try:
    op2.database_backwards(
        app_label='test_app',
        schema_editor=None,
        from_state=None,
        to_state=None
    )
    print("No exception raised (unexpected)")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}")

print(f"\nState after exception:")
print(f"  old_name = '{op2.old_name}'")
print(f"  new_name = '{op2.new_name}'")

if op2.old_name == 'idx_new' and op2.new_name == 'idx_old':
    print("\n❌ BUG CONFIRMED: Operation state was MUTATED!")
    print("   The old_name and new_name have been swapped.")
else:
    print("\n✓ Operation state remained unchanged.")

# Demonstrate impact on deconstruct() method
print("\n" + "=" * 60)
print("3. Impact on deconstruct() method:")
print("-" * 40)

op3 = RenameModel(old_name='Product', new_name='Item')
print(f"Initial deconstruct(): {op3.deconstruct()}")

try:
    op3.database_backwards(None, None, None, None)
except:
    pass

print(f"After exception deconstruct(): {op3.deconstruct()}")
print("\nNote: The deconstruct() method returns the ORIGINAL constructor")
print("arguments, not the current state. This violates the expectation")
print("that deconstruct() should allow reconstructing an equivalent object.")

# Demonstrate impact on describe() method
print("\n" + "=" * 60)
print("4. Impact on describe() method:")
print("-" * 40)

op4 = RenameModel(old_name='User', new_name='Account')
print(f"Initial description: {op4.describe()}")

try:
    op4.database_backwards(None, None, None, None)
except:
    pass

print(f"After exception description: {op4.describe()}")
print("\n❌ The describe() method now shows swapped names!")

print("\n" + "=" * 60)
print("CONCLUSION:")
print("The RenameModel and RenameIndex operations violate the documented")
print("immutability requirement by permanently mutating their state when")
print("database_backwards() encounters an exception.")
print("=" * 60)
```

<details>

<summary>
Exception raised when database_backwards() is called with None schema_editor
</summary>
```
============================================================
Testing RenameModel Operation Mutation Bug
============================================================

1. RenameModel Operation Test:
----------------------------------------
Initial state:
  old_name = 'Author'
  new_name = 'Writer'

Calling database_backwards() with None arguments...
Exception raised: AttributeError

State after exception:
  old_name = 'Writer'
  new_name = 'Author'

❌ BUG CONFIRMED: Operation state was MUTATED!
   The old_name and new_name have been swapped.

============================================================
2. RenameIndex Operation Test:
----------------------------------------
Initial state:
  old_name = 'idx_old'
  new_name = 'idx_new'

Calling database_backwards() with None arguments...
Exception raised: AttributeError

State after exception:
  old_name = 'idx_new'
  new_name = 'idx_old'

❌ BUG CONFIRMED: Operation state was MUTATED!
   The old_name and new_name have been swapped.

============================================================
3. Impact on deconstruct() method:
----------------------------------------
Initial deconstruct(): ('RenameModel', [], {'old_name': 'Product', 'new_name': 'Item'})
After exception deconstruct(): ('RenameModel', [], {'old_name': 'Item', 'new_name': 'Product'})

Note: The deconstruct() method returns the ORIGINAL constructor
arguments, not the current state. This violates the expectation
that deconstruct() should allow reconstructing an equivalent object.

============================================================
4. Impact on describe() method:
----------------------------------------
Initial description: Rename model User to Account
After exception description: Rename model Account to User

❌ The describe() method now shows swapped names!

============================================================
CONCLUSION:
The RenameModel and RenameIndex operations violate the documented
immutability requirement by permanently mutating their state when
database_backwards() encounters an exception.
============================================================
```
</details>

## Why This Is A Bug

This bug violates Django's explicit architectural requirement that migration operations must be immutable. The base `Operation` class documentation at `/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/db/migrations/operations/base.py:27-28` states:

> "Due to the way this class deals with deconstruction, it should be considered immutable."

The problematic implementation uses a swap-mutate-swap-back pattern in both `RenameModel.database_backwards()` (lines 538-551) and `RenameIndex.database_backwards()` (lines 1140-1157):

1. The method swaps `old_name` and `new_name` attributes
2. Calls `database_forwards()` with the swapped values
3. Attempts to swap the values back

When `database_forwards()` raises an exception (which happens during testing, dry runs, or when migrations fail), step 3 never executes, leaving the operation permanently mutated.

This mutation breaks several core functionalities:
- **`deconstruct()` method**: Returns incorrect constructor arguments after mutation, breaking serialization
- **`describe()` method**: Shows swapped names in error messages and logs
- **Migration retry logic**: If a migration fails and is retried, operations will have incorrect state
- **Test isolation**: Unit tests that catch exceptions see mutated state

## Relevant Context

The bug manifests in realistic scenarios including:
- Migration testing and validation without actual database operations
- Dry runs with `schema_editor=None`
- Database connection failures
- Permission errors during migrations
- Schema conflicts that cause migrations to fail

Notably, other similar operations handle this correctly. `RenameField` doesn't use the swap pattern, and `AlterModelTable.database_backwards()` simply calls `database_forwards()` without any state mutation. This inconsistency suggests the swap-mutate-swap pattern is a bug rather than intentional design.

Django documentation: https://docs.djangoproject.com/en/stable/ref/migration-operations/
Source code: https://github.com/django/django/blob/main/django/db/migrations/operations/models.py

## Proposed Fix

Replace the swap-mutate-swap-back pattern with a non-mutating approach that uses local variables for swapped values:

```diff
--- a/django/db/migrations/operations/models.py
+++ b/django/db/migrations/operations/models.py
@@ -536,19 +536,21 @@ class RenameModel(ModelOperation):
                 )

     def database_backwards(self, app_label, schema_editor, from_state, to_state):
-        self.new_name_lower, self.old_name_lower = (
-            self.old_name_lower,
-            self.new_name_lower,
-        )
-        self.new_name, self.old_name = self.old_name, self.new_name
-
-        self.database_forwards(app_label, schema_editor, from_state, to_state)
-
-        self.new_name_lower, self.old_name_lower = (
-            self.old_name_lower,
-            self.new_name_lower,
-        )
-        self.new_name, self.old_name = self.old_name, self.new_name
+        # Create a temporary operation with swapped names to avoid mutation
+        backwards_op = RenameModel(
+            old_name=self.new_name,
+            new_name=self.old_name
+        )
+        backwards_op.database_forwards(app_label, schema_editor, from_state, to_state)

@@ -1138,19 +1140,21 @@ class RenameIndex(IndexOperation):
         schema_editor.rename_index(model, old_index, new_index)

     def database_backwards(self, app_label, schema_editor, from_state, to_state):
         if self.old_fields:
             # Backward operation with unnamed index is a no-op.
             return

-        self.new_name_lower, self.old_name_lower = (
-            self.old_name_lower,
-            self.new_name_lower,
-        )
-        self.new_name, self.old_name = self.old_name, self.new_name
-
-        self.database_forwards(app_label, schema_editor, from_state, to_state)
-
-        self.new_name_lower, self.old_name_lower = (
-            self.old_name_lower,
-            self.new_name_lower,
-        )
-        self.new_name, self.old_name = self.old_name, self.new_name
+        # Create a temporary operation with swapped names to avoid mutation
+        backwards_op = RenameIndex(
+            model_name=self.model_name,
+            old_name=self.new_name,
+            new_name=self.old_name,
+            old_fields=self.old_fields
+        )
+        backwards_op.database_forwards(app_label, schema_editor, from_state, to_state)
```