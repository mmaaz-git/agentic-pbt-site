# Bug Report: Django Migrations Operations Immutability Contract Violation

**Target**: `django.db.migrations.operations.RenameModel` and `django.db.migrations.operations.RenameIndex`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`RenameModel` and `RenameIndex` operations violate the immutability contract by mutating instance attributes during `database_backwards()` execution, causing the operation's state to be permanently altered.

## Property-Based Test

```python
import sys
# Add Django to the path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, assume
from django.db.migrations.operations import RenameModel

@given(
    old_name=st.text(min_size=1, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))),
    new_name=st.text(min_size=1, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))),
)
def test_renamemodel_immutability(old_name, new_name):
    assume(old_name != new_name)

    op = RenameModel(old_name=old_name, new_name=new_name)

    # Store the original values
    original_old_name = op.old_name
    original_new_name = op.new_name
    original_old_name_lower = op.old_name_lower
    original_new_name_lower = op.new_name_lower

    from django.db.migrations.state import ProjectState
    state = ProjectState()

    # Call database_backwards - it might fail due to missing schema_editor,
    # but we're testing whether it mutates the instance
    try:
        op.database_backwards('test_app', None, state, state)
    except:
        pass

    # Check that the instance attributes were not mutated
    assert op.old_name == original_old_name, f"old_name was mutated from {original_old_name} to {op.old_name}"
    assert op.new_name == original_new_name, f"new_name was mutated from {original_new_name} to {op.new_name}"
    assert op.old_name_lower == original_old_name_lower, f"old_name_lower was mutated"
    assert op.new_name_lower == original_new_name_lower, f"new_name_lower was mutated"

if __name__ == "__main__":
    # Run the test
    test_renamemodel_immutability()
```

<details>

<summary>
**Failing input**: `old_name='A', new_name='d'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 41, in <module>
    test_renamemodel_immutability()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 9, in test_renamemodel_immutability
    old_name=st.text(min_size=1, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))),
               ^^^
  File "/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/56/hypo.py", line 34, in test_renamemodel_immutability
    assert op.old_name == original_old_name, f"old_name was mutated from {original_old_name} to {op.old_name}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: old_name was mutated from A to d
Falsifying example: test_renamemodel_immutability(
    old_name='A',
    new_name='d',
)
```
</details>

## Reproducing the Bug

```python
import sys
# Add Django to the path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.db.migrations.operations import RenameModel
from django.db.migrations.state import ProjectState

# Create a RenameModel operation
op = RenameModel(old_name="User", new_name="Person")

print(f"Before database_backwards:")
print(f"  op.old_name = {op.old_name}")
print(f"  op.new_name = {op.new_name}")
print(f"  op.old_name_lower = {op.old_name_lower}")
print(f"  op.new_name_lower = {op.new_name_lower}")

# Create a project state
state = ProjectState()

# Call database_backwards
try:
    op.database_backwards('test_app', None, state, state)
except:
    # It may fail due to missing schema_editor, but that's not the issue we're testing
    pass

print(f"\nAfter database_backwards:")
print(f"  op.old_name = {op.old_name}")
print(f"  op.new_name = {op.new_name}")
print(f"  op.old_name_lower = {op.old_name_lower}")
print(f"  op.new_name_lower = {op.new_name_lower}")

print("\n=== Mutation detected! ===")
print("The instance attributes were mutated by database_backwards,")
print("violating the immutability contract documented in the Operation base class.")
```

<details>

<summary>
Instance attributes permanently swapped after database_backwards() call
</summary>
```
Before database_backwards:
  op.old_name = User
  op.new_name = Person
  op.old_name_lower = user
  op.new_name_lower = person

After database_backwards:
  op.old_name = Person
  op.new_name = User
  op.old_name_lower = person
  op.new_name_lower = user

=== Mutation detected! ===
The instance attributes were mutated by database_backwards,
violating the immutability contract documented in the Operation base class.
```
</details>

## Why This Is A Bug

The `Operation` base class documentation at `django/db/migrations/operations/base.py:27-28` explicitly states:

> "Due to the way this class deals with deconstruction, it should be considered immutable."

However, both `RenameModel.database_backwards()` and `RenameIndex.database_backwards()` violate this immutability contract by directly mutating instance attributes.

In `RenameModel.database_backwards()` (lines 538-551):
```python
def database_backwards(self, app_label, schema_editor, from_state, to_state):
    self.new_name_lower, self.old_name_lower = (
        self.old_name_lower,
        self.new_name_lower,
    )
    self.new_name, self.old_name = self.old_name, self.new_name

    self.database_forwards(app_label, schema_editor, from_state, to_state)

    # Swaps them back
    self.new_name_lower, self.old_name_lower = (
        self.old_name_lower,
        self.new_name_lower,
    )
    self.new_name, self.old_name = self.old_name, self.new_name
```

This implementation has several critical issues:

1. **Exception Safety Violation**: If `database_forwards()` raises an exception after the first swap but before the second swap, the operation instance remains in a corrupted state with swapped attributes.

2. **Thread Safety Violation**: If the same operation instance is used concurrently in different threads (e.g., during parallel migration execution), threads could observe inconsistent states.

3. **Reusability Violation**: The operation cannot be safely cached or reused since its state gets mutated during execution.

4. **Deconstruction Integrity Violation**: The `deconstruct()` method relies on the original constructor arguments. If attributes are swapped when an exception occurs, `deconstruct()` returns incorrect values.

The same issue exists in `RenameIndex.database_backwards()` (lines 1145-1157).

## Relevant Context

The immutability contract is essential for Django's migration system because:
- Operations are stored in migration files and may be reused during squashing or optimization
- The migration optimizer relies on operations maintaining consistent state
- Operations use `_constructor_args` to implement `deconstruct()`, assuming attributes match constructor arguments
- Django's migration executor may retry operations or roll them back

Related Django documentation: https://docs.djangoproject.com/en/stable/topics/migrations/#migration-operations

Code locations:
- Base class: `/django/db/migrations/operations/base.py`
- RenameModel: `/django/db/migrations/operations/models.py:462-581`
- RenameIndex: `/django/db/migrations/operations/models.py:1035-1194`

## Proposed Fix

```diff
--- a/django/db/migrations/operations/models.py
+++ b/django/db/migrations/operations/models.py
@@ -537,18 +537,24 @@ class RenameModel(ModelOperation):
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
+        # Create a new operation with swapped names to avoid mutation
+        reverse_op = RenameModel(
+            old_name=self.new_name,
+            new_name=self.old_name
+        )
+        reverse_op.database_forwards(app_label, schema_editor, from_state, to_state)


@@ -1140,18 +1146,13 @@ class RenameIndex(IndexOperation):
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
+        # Create a new operation with swapped names to avoid mutation
+        reverse_op = RenameIndex(
+            model_name=self.model_name,
+            old_name=self.new_name,
+            new_name=self.old_name
+        )
+        reverse_op.database_forwards(app_label, schema_editor, from_state, to_state)
```