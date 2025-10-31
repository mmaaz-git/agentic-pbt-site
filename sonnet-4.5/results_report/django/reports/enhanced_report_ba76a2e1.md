# Bug Report: django.db.migrations.operations.AddIndex Violates Immutability Contract in reduce() Method

**Target**: `django.db.migrations.operations.AddIndex.reduce()`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `AddIndex.reduce()` method directly mutates the operation's `index.name` attribute, violating the documented immutability contract that all migration operations must follow.

## Property-Based Test

```python
import django
from django.conf import settings

if not settings.configured:
    settings.configure(
        DATABASES={'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': ':memory:'}},
        INSTALLED_APPS=['django.contrib.contenttypes', 'django.contrib.auth'],
        SECRET_KEY='test',
    )
    django.setup()

from hypothesis import given, settings as hypo_settings, strategies as st
from django.db.models import Index
from django.db.migrations.operations import AddIndex, RenameIndex

@given(
    model_name=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("Lu", "Ll"))),
    index_name=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("Lu", "Ll"))),
    new_index_name=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("Lu", "Ll"))),
)
@hypo_settings(max_examples=1000)
def test_add_index_immutability_in_reduce(model_name, index_name, new_index_name):
    index = Index(fields=["id"], name=index_name)
    op = AddIndex(model_name=model_name, index=index)

    original_index_name = op.index.name

    rename_op = RenameIndex(model_name=model_name, old_name=index_name, new_name=new_index_name)
    result = op.reduce(rename_op, "app")

    assert op.index.name == original_index_name, f"Index name was mutated from {original_index_name} to {op.index.name}"

if __name__ == "__main__":
    # Run the test
    test_add_index_immutability_in_reduce()
```

<details>

<summary>
**Failing input**: `model_name='A', index_name='A', new_index_name='AA'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 35, in <module>
    test_add_index_immutability_in_reduce()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 17, in test_add_index_immutability_in_reduce
    model_name=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("Lu", "Ll"))),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 31, in test_add_index_immutability_in_reduce
    assert op.index.name == original_index_name, f"Index name was mutated from {original_index_name} to {op.index.name}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Index name was mutated from A to AA
Falsifying example: test_add_index_immutability_in_reduce(
    model_name='A',  # or any other generated value
    index_name='A',  # or any other generated value
    new_index_name='AA',
)
```
</details>

## Reproducing the Bug

```python
import django
from django.conf import settings

if not settings.configured:
    settings.configure(
        DATABASES={'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': ':memory:'}},
        INSTALLED_APPS=['django.contrib.contenttypes', 'django.contrib.auth'],
        SECRET_KEY='test',
    )
    django.setup()

from django.db.models import Index
from django.db.migrations.operations import AddIndex, RenameIndex

# Create an AddIndex operation with an index
index = Index(fields=["id"], name="old_index")
add_op = AddIndex(model_name="MyModel", index=index)

# Store original index name for verification
original_name = add_op.index.name
print(f"Before reduce: {add_op.index.name}")

# Create a RenameIndex operation
rename_op = RenameIndex(model_name="MyModel", old_name="old_index", new_name="new_index")

# Call reduce, which should NOT mutate the original operation
result = add_op.reduce(rename_op, "app")

print(f"After reduce: {add_op.index.name}")

# Verify the mutation
if add_op.index.name != original_name:
    print(f"ERROR: Index name was mutated from {original_name} to {add_op.index.name}")
    print("This violates the immutability contract of migration operations.")
else:
    print("SUCCESS: Index name remained unchanged.")
```

<details>

<summary>
Demonstrates mutation of index.name from "old_index" to "new_index"
</summary>
```
Before reduce: old_index
After reduce: new_index
ERROR: Index name was mutated from old_index to new_index
This violates the immutability contract of migration operations.
```
</details>

## Why This Is A Bug

This bug violates Django's explicit architectural contract for migration operations. The `Operation` base class at `/django/db/migrations/operations/base.py` lines 27-28 clearly states:

> "Due to the way this class deals with deconstruction, it should be considered immutable."

The `AddIndex.reduce()` method at line 993 of `/django/db/migrations/operations/models.py` directly violates this contract:

```python
if isinstance(operation, RenameIndex) and self.index.name == operation.old_name:
    self.index.name = operation.new_name  # MUTATION!
    return [self.__class__(model_name=self.model_name, index=self.index)]
```

This mutation causes several problems:

1. **Reuse Issues**: The Django migration optimizer runs multiple optimization iterations and may reuse operation instances. When an `AddIndex` operation is mutated during one pass, subsequent uses see the mutated state rather than the original.

2. **Order-Dependent Behavior**: The operation's final state depends on the order and number of times `reduce()` is called, making the optimization process unpredictable and harder to debug.

3. **Contract Violation**: The `reduce()` method documentation (lines 150-154 of base.py) states it should "Return either a list of operations...or a boolean", implying new operations should be created, not existing ones mutated.

4. **Optimizer Instability**: The migration optimizer requires operations to be stable across multiple passes. Mutation breaks this stability guarantee.

## Relevant Context

The Django `Index` class already provides a `clone()` method (confirmed to exist in the current Django version) specifically designed for creating copies when modifications are needed. This makes the fix straightforward - the code should use `clone()` instead of mutating the original.

The migration optimizer is a critical component that runs automatically when migrations are created, affecting every Django project that uses database migrations. While the bug may not always manifest visibly (since the mutated operation often still works correctly), it can lead to subtle issues during complex migration sequences or when custom migration optimizers are used.

Documentation references:
- Operation immutability: `/django/db/migrations/operations/base.py:27-28`
- Reduce method contract: `/django/db/migrations/operations/base.py:150-154`
- Problematic code: `/django/db/migrations/operations/models.py:993`

## Proposed Fix

```diff
--- a/django/db/migrations/operations/models.py
+++ b/django/db/migrations/operations/models.py
@@ -990,8 +990,9 @@ class AddIndex(IndexOperation):
         if isinstance(operation, RemoveIndex) and self.index.name == operation.name:
             return []
         if isinstance(operation, RenameIndex) and self.index.name == operation.old_name:
-            self.index.name = operation.new_name
-            return [self.__class__(model_name=self.model_name, index=self.index)]
+            renamed_index = self.index.clone()
+            renamed_index.name = operation.new_name
+            return [self.__class__(model_name=self.model_name, index=renamed_index)]
         return super().reduce(operation, app_label)
```