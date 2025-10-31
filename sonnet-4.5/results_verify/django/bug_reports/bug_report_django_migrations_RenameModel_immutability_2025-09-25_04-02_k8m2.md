# Bug Report: RenameModel and RenameIndex Violate Immutability on Exception

**Target**: `django.db.migrations.operations.models.RenameModel` and `django.db.migrations.operations.models.RenameIndex`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `database_backwards()` methods in `RenameModel` and `RenameIndex` mutate the operation's state by swapping name attributes, but fail to restore the original state if `database_forwards()` raises an exception. This violates the immutability property documented in `base.py:28`: "Due to the way this class deals with deconstruction, it should be considered immutable."

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.db.migrations.operations import RenameModel

@given(
    st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))),
    st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll')))
)
def test_rename_model_immutability_after_exception(old_name, new_name):
    from hypothesis import assume
    assume(old_name != new_name and old_name.isidentifier() and new_name.isidentifier())

    op = RenameModel(old_name, new_name)

    _ = op.old_name_lower
    _ = op.new_name_lower

    original_old_name = op.old_name
    original_new_name = op.new_name
    original_old_name_lower = op.old_name_lower
    original_new_name_lower = op.new_name_lower

    try:
        op.database_backwards("app", None, None, None)
    except:
        pass

    assert op.old_name == original_old_name
    assert op.new_name == original_new_name
    assert op.old_name_lower == original_old_name_lower
    assert op.new_name_lower == original_new_name_lower
```

**Failing input**: `old_name="Article"`, `new_name="Post"`

## Reproducing the Bug

```python
from django.db.migrations.operations import RenameModel

op = RenameModel("Article", "Post")

_ = op.old_name_lower
_ = op.new_name_lower

print(f"Before: old_name={op.old_name!r}, new_name={op.new_name!r}")
print(f"        old_name_lower={op.old_name_lower!r}, new_name_lower={op.new_name_lower!r}")

try:
    op.database_backwards("myapp", None, None, None)
except AttributeError:
    pass

print(f"After:  old_name={op.old_name!r}, new_name={op.new_name!r}")
print(f"        old_name_lower={op.old_name_lower!r}, new_name_lower={op.new_name_lower!r}")
```

**Expected output**:
```
Before: old_name='Article', new_name='Post'
        old_name_lower='article', new_name_lower='post'
After:  old_name='Article', new_name='Post'
        old_name_lower='article', new_name_lower='post'
```

**Actual output**:
```
Before: old_name='Article', new_name='Post'
        old_name_lower='article', new_name_lower='post'
After:  old_name='Post', new_name='Article'
        old_name_lower='post', new_name_lower='article'
```

## Why This Is A Bug

The `database_backwards()` implementation at `models.py:539-551` swaps the operation's attributes to reuse `database_forwards()`:

```python
def database_backwards(self, app_label, schema_editor, from_state, to_state):
    self.new_name_lower, self.old_name_lower = (
        self.old_name_lower,
        self.new_name_lower,
    )
    self.new_name, self.old_name = self.old_name, self.new_name

    self.database_forwards(app_label, schema_editor, from_state, to_state)  # Can raise!

    self.new_name_lower, self.old_name_lower = (
        self.old_name_lower,
        self.new_name_lower,
    )
    self.new_name, self.old_name = self.old_name, self.new_name
```

If `database_forwards()` raises any exception, the swap-back code is never executed, leaving the operation permanently mutated.

This violates:
1. The immutability contract stated in `base.py:28`
2. The expectation that operation objects remain consistent after method calls
3. Proper exception safety - the operation should be left in a valid state

The same bug exists in `RenameIndex.database_backwards()` at `models.py:1140-1157`.

## Fix

Use a try-finally block to ensure the swap-back always happens:

```diff
diff --git a/django/db/migrations/operations/models.py b/django/db/migrations/operations/models.py
--- a/django/db/migrations/operations/models.py
+++ b/django/db/migrations/operations/models.py
@@ -538,16 +538,18 @@ class RenameModel(ModelOperation):
     def database_backwards(self, app_label, schema_editor, from_state, to_state):
         self.new_name_lower, self.old_name_lower = (
             self.old_name_lower,
             self.new_name_lower,
         )
         self.new_name, self.old_name = self.old_name, self.new_name

-        self.database_forwards(app_label, schema_editor, from_state, to_state)
-
-        self.new_name_lower, self.old_name_lower = (
-            self.old_name_lower,
-            self.new_name_lower,
-        )
-        self.new_name, self.old_name = self.old_name, self.new_name
+        try:
+            self.database_forwards(app_label, schema_editor, from_state, to_state)
+        finally:
+            self.new_name_lower, self.old_name_lower = (
+                self.old_name_lower,
+                self.new_name_lower,
+            )
+            self.new_name, self.old_name = self.old_name, self.new_name
```

Apply the same fix to `RenameIndex.database_backwards()` at lines 1140-1157.