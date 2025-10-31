# Bug Report: Django Migrations Operations Immutability Violation

**Target**: `django.db.migrations.operations.RenameModel` and `django.db.migrations.operations.RenameIndex`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`RenameModel` and `RenameIndex` operations violate the immutability contract documented in the base `Operation` class by mutating instance attributes in `database_backwards()`. This creates potential issues with thread safety, exception handling, and operation reuse.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from django.db.migrations.operations import RenameModel

@given(
    old_name=st.text(min_size=1, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))),
    new_name=st.text(min_size=1, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))),
)
def test_renamemodel_immutability(old_name, new_name):
    assume(old_name != new_name)

    op = RenameModel(old_name=old_name, new_name=new_name)

    original_old_name = op.old_name
    original_new_name = op.new_name

    from django.db.migrations.state import ProjectState
    state = ProjectState()

    try:
        op.database_backwards('test_app', None, state, state)
    except:
        pass

    assert op.old_name == original_old_name
    assert op.new_name == original_new_name
```

**Failing input**: Any valid model names, e.g., `old_name="User"`, `new_name="Person"`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/path/to/django')

from django.db.migrations.operations import RenameModel
from django.db.migrations.state import ProjectState

op = RenameModel(old_name="User", new_name="Person")

print(f"Before: old_name={op.old_name}, new_name={op.new_name}")

state = ProjectState()
op.database_backwards('test_app', None, state, state)

print(f"After: old_name={op.old_name}, new_name={op.new_name}")
```

## Why This Is A Bug

The `Operation` base class documentation at `base.py:27-28` explicitly states:

> "Due to the way this class deals with deconstruction, it should be considered immutable."

However, `RenameModel.database_backwards()` (lines 539-551) and `RenameIndex.database_backwards()` (lines 1145-1157) mutate instance attributes:

```python
def database_backwards(self, app_label, schema_editor, from_state, to_state):
    self.new_name_lower, self.old_name_lower = (
        self.old_name_lower,
        self.new_name_lower,
    )
    self.new_name, self.old_name = self.old_name, self.new_name
    # ...
```

This violates the immutability contract and creates several issues:

1. **Exception safety**: If `database_forwards()` raises an exception, attributes remain swapped
2. **Thread safety**: Concurrent execution could see inconsistent state
3. **Reusability**: The operation cannot be safely reused or cached
4. **Contract violation**: Breaks documented behavior that other code may depend on

## Fix

```diff
--- a/django/db/migrations/operations/models.py
+++ b/django/db/migrations/operations/models.py
@@ -536,18 +536,11 @@ class RenameModel(ModelOperation):
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
+        new_model = from_state.apps.get_model(app_label, self.old_name)
+        if self.allow_migrate_model(schema_editor.connection.alias, new_model):
+            old_model = to_state.apps.get_model(app_label, self.new_name)
+            schema_editor.alter_db_table(
+                new_model,
+                old_model._meta.db_table,
+                new_model._meta.db_table,
+            )
+            # Handle related objects...

--- a/django/db/migrations/operations/models.py
+++ b/django/db/migrations/operations/models.py
@@ -1140,18 +1140,10 @@ class RenameIndex(IndexOperation):
     def database_backwards(self, app_label, schema_editor, from_state, to_state):
         if self.old_fields:
-            # Backward operation with unnamed index is a no-op.
             return
-
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
+
+        model = from_state.apps.get_model(app_label, self.model_name)
+        if not self.allow_migrate_model(schema_editor.connection.alias, model):
+            return
+        # Implement backwards logic without mutation
```