# Bug Report: django.db.migrations.operations.models.RenameModel Violates Immutability Contract on Exception

**Target**: `django.db.migrations.operations.models.RenameModel.database_backwards`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `database_backwards()` method in `RenameModel` mutates the operation's internal state by swapping name attributes but fails to restore them if an exception occurs, violating the documented immutability contract that states migration operations "should be considered immutable."

## Property-Based Test

```python
#!/usr/bin/env python3
"""Hypothesis property test for RenameModel immutability."""

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

    # Force cached properties to be computed
    _ = op.old_name_lower
    _ = op.new_name_lower

    # Store original values
    original_old_name = op.old_name
    original_new_name = op.new_name
    original_old_name_lower = op.old_name_lower
    original_new_name_lower = op.new_name_lower

    # Try database_backwards, which will fail with None schema_editor
    try:
        op.database_backwards("app", None, None, None)
    except:
        pass

    # Check that the operation remains unchanged (immutable)
    assert op.old_name == original_old_name, f"old_name changed: {original_old_name!r} -> {op.old_name!r}"
    assert op.new_name == original_new_name, f"new_name changed: {original_new_name!r} -> {op.new_name!r}"
    assert op.old_name_lower == original_old_name_lower, f"old_name_lower changed: {original_old_name_lower!r} -> {op.old_name_lower!r}"
    assert op.new_name_lower == original_new_name_lower, f"new_name_lower changed: {original_new_name_lower!r} -> {op.new_name_lower!r}"

if __name__ == "__main__":
    # Run the property test
    test_rename_model_immutability_after_exception()
    print("Test passed!")
```

<details>

<summary>
**Failing input**: `old_name='A', new_name='Ü'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 41, in <module>
    test_rename_model_immutability_after_exception()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 8, in test_rename_model_immutability_after_exception
    st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/33/hypo.py", line 34, in test_rename_model_immutability_after_exception
    assert op.old_name == original_old_name, f"old_name changed: {original_old_name!r} -> {op.old_name!r}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: old_name changed: 'A' -> 'Ü'
Falsifying example: test_rename_model_immutability_after_exception(
    old_name='A',
    new_name='Ü',
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction of RenameModel immutability bug."""

from django.db.migrations.operations import RenameModel

# Create a RenameModel operation
op = RenameModel("Article", "Post")

# Force the cached properties to be computed before the test
_ = op.old_name_lower
_ = op.new_name_lower

# Print initial state
print(f"Before: old_name={op.old_name!r}, new_name={op.new_name!r}")
print(f"        old_name_lower={op.old_name_lower!r}, new_name_lower={op.new_name_lower!r}")

# Try to call database_backwards with None schema_editor (causes AttributeError)
try:
    op.database_backwards("myapp", None, None, None)
except AttributeError as e:
    print(f"\nException raised: {e}")

# Print state after exception
print(f"\nAfter:  old_name={op.old_name!r}, new_name={op.new_name!r}")
print(f"        old_name_lower={op.old_name_lower!r}, new_name_lower={op.new_name_lower!r}")

# Check if the state was properly restored (it should be unchanged)
if op.old_name == "Article" and op.new_name == "Post":
    print("\n✓ Operation state correctly preserved after exception")
else:
    print("\n✗ BUG: Operation state was mutated after exception!")
    print("  Expected: old_name='Article', new_name='Post'")
    print(f"  Got:      old_name={op.old_name!r}, new_name={op.new_name!r}")
```

<details>

<summary>
AttributeError exception causes permanent mutation of operation state
</summary>
```
Before: old_name='Article', new_name='Post'
        old_name_lower='article', new_name_lower='post'

Exception raised: 'NoneType' object has no attribute 'apps'

After:  old_name='Post', new_name='Article'
        old_name_lower='post', new_name_lower='article'

✗ BUG: Operation state was mutated after exception!
  Expected: old_name='Article', new_name='Post'
  Got:      old_name='Post', new_name='Article'
```
</details>

## Why This Is A Bug

This violates the explicit immutability contract documented in Django's base `Operation` class at `/django/db/migrations/operations/base.py` lines 27-28: "Due to the way this class deals with deconstruction, it should be considered immutable."

The bug occurs because `database_backwards()` at lines 538-551 swaps the operation's name attributes to reuse `database_forwards()`, but when `database_forwards()` raises an exception (line 545), the restoration code (lines 547-551) never executes. This leaves the operation permanently mutated with swapped old_name/new_name values.

This violates three key principles:
1. **Immutability Contract**: Operations must remain unchanged to support reliable deconstruction/reconstruction for serialization
2. **Exception Safety**: Objects should remain in a valid, consistent state after method failures
3. **Reusability**: Migration operations should be safely reusable across different contexts without side effects

The same bug exists in `RenameIndex.database_backwards()` at lines 1140-1157.

## Relevant Context

The immutability requirement exists because Django's migration system uses a deconstruction pattern (via the `deconstruct()` method) to serialize and reconstruct operation objects. A mutated operation would produce incorrect results when deconstructed and reconstructed.

While the bug only manifests when exceptions occur during backward migrations (e.g., when schema_editor is None or when database operations fail), it represents a fundamental violation of the documented contract and proper exception handling principles.

Django documentation: https://docs.djangoproject.com/en/stable/ref/migration-operations/
Source code: https://github.com/django/django/blob/main/django/db/migrations/operations/models.py

## Proposed Fix

```diff
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

     def references_model(self, name, app_label):
         return (
@@ -1140,16 +1142,18 @@ class RenameIndex(IndexOperation):
     def database_backwards(self, app_label, schema_editor, from_state, to_state):
         self.old_name, self.new_name = self.new_name, self.old_name

-        self.database_forwards(app_label, schema_editor, from_state, to_state)
-
-        self.new_name, self.old_name = self.old_name, self.new_name
+        try:
+            self.database_forwards(app_label, schema_editor, from_state, to_state)
+        finally:
+            self.new_name, self.old_name = self.old_name, self.new_name
```