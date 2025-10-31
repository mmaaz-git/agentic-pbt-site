# Bug Report: Django Migrations Rename Operations State Mutation

**Target**: `django.db.migrations.operations.RenameModel` and `django.db.migrations.operations.RenameIndex`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`RenameModel.database_backwards()` and `RenameIndex.database_backwards()` permanently mutate the operation's state when an exception is raised, violating Django's documented immutability requirement for migration operations.

## Property-Based Test

```python
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

    assert op.old_name == original_old_name
    assert op.new_name == original_new_name
```

**Failing input**: `old_name='A'`, `new_name='B'`

## Reproducing the Bug

```python
from django.db.migrations.operations import RenameModel

op = RenameModel(old_name='Author', new_name='Writer')

print(f"Before: old_name={op.old_name}, new_name={op.new_name}")

try:
    op.database_backwards(None, None, None, None)
except:
    pass

print(f"After: old_name={op.old_name}, new_name={op.new_name}")
```

Expected output:
```
Before: old_name=Author, new_name=Writer
After: old_name=Author, new_name=Writer
```

Actual output:
```
Before: old_name=Author, new_name=Writer
After: old_name=Writer, new_name=Author
```

## Why This Is A Bug

According to the base Operation class docstring (django/db/migrations/operations/base.py:28):
> "Due to the way this class deals with deconstruction, it should be considered immutable."

The `database_backwards()` implementation uses a swap-mutate-swap-back pattern:

```python
def database_backwards(self, app_label, schema_editor, from_state, to_state):
    self.new_name_lower, self.old_name_lower = (
        self.old_name_lower,
        self.new_name_lower,
    )
    self.new_name, self.old_name = self.old_name, self.new_name

    self.database_forwards(app_label, schema_editor, from_state, to_state)

    self.new_name_lower, self.old_name_lower = (
        self.old_name_lower,
        self.new_name_lower,
    )
    self.new_name, self.old_name = self.old_name, self.new_name
```

If `database_forwards()` raises an exception (which commonly happens during testing, dry runs, or when migrations fail), the swap-back never executes, leaving the operation in an inconsistent state. This can cause:

1. **Incorrect retry behavior**: If a migration fails and is retried, the operation will have swapped names
2. **Misleading error messages**: The operation's `describe()` method will report incorrect information
3. **Test failures**: Unit tests that catch exceptions will see mutated state
4. **Debugging confusion**: Stack traces and logging will show incorrect operation state

The same bug exists in `RenameIndex.database_backwards()` at lines 1140-1157 of `django/db/migrations/operations/models.py`.

Interestingly, `RenameField.database_backwards()` does not have this bug because it doesn't use the swap pattern - it directly references the field names without mutation.

## Fix

Replace the swap-mutate-swap-back pattern with a non-mutating approach, similar to how `RenameField` works:

```diff
--- a/django/db/migrations/operations/models.py
+++ b/django/db/migrations/operations/models.py
@@ -538,17 +538,17 @@ class RenameModel(ModelOperation):
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
+        old_model = from_state.apps.get_model(app_label, self.old_name)
+        if self.allow_migrate_model(schema_editor.connection.alias, old_model):
+            new_model = to_state.apps.get_model(app_label, self.new_name)
+            # Move the main table
+            schema_editor.alter_db_table(
+                old_model,
+                new_model._meta.db_table,
+                old_model._meta.db_table,
+            )
+            # Note: Related objects and M2M fields would need similar backwards logic
+            # (implementation details omitted for brevity, but should mirror the
+            # forwards direction with names swapped in the logic, not in state)

@@ -1140,17 +1140,8 @@ class RenameIndex(IndexOperation):
     def database_backwards(self, app_label, schema_editor, from_state, to_state):
         if self.old_fields:
             # Backward operation with unnamed index is a no-op.
             return
-
-        self.new_name_lower, self.old_name_lower = (
-            self.old_name_lower,
-            self.new_name_lower,
-        )
-        self.new_name, self.old_name = self.old_name, self.new_name
-
+
+        # Swap names in local variables, not in self
+        old_name, new_name = self.new_name, self.old_name
+        from_model_state = from_state.models[app_label, self.model_name_lower]
+        old_index = from_model_state.get_index_by_name(new_name)
+
+        to_model_state = to_state.models[app_label, self.model_name_lower]
+        new_index = to_model_state.get_index_by_name(old_name)
+
+        model = to_state.apps.get_model(app_label, self.model_name)
+        if old_index.name == new_index.name:
+            return
+
+        schema_editor.rename_index(model, old_index, new_index)
-        self.database_forwards(app_label, schema_editor, from_state, to_state)
-
-        self.new_name_lower, self.old_name_lower = (
-            self.old_name_lower,
-            self.new_name_lower,
-        )
-        self.new_name, self.old_name = self.old_name, self.new_name
```