# Bug Report: AddIndex.reduce() Mutates Original Index Object

**Target**: `django.db.migrations.operations.AddIndex.reduce`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `AddIndex.reduce()` method mutates the original operation's index object when reducing with a `RenameIndex` operation. This violates the immutability contract and can lead to unexpected state changes.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from django.db.migrations.operations import AddIndex, RenameIndex
from django.db import models

@st.composite
def model_names(draw):
    chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    name = draw(st.text(alphabet=chars, min_size=1, max_size=20))
    assume(name.isidentifier())
    return name

@st.composite
def index_names(draw):
    chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_'
    name = draw(st.text(alphabet=chars, min_size=1, max_size=20))
    assume(name.isidentifier())
    return name

@given(model_names(), index_names(), index_names())
def test_add_index_reduce_immutability(model_name, old_idx_name, new_idx_name):
    assume(old_idx_name != new_idx_name)

    index = models.Index(fields=['name'], name=old_idx_name)
    add_op = AddIndex(model_name=model_name, index=index)
    rename_op = RenameIndex(model_name=model_name, old_name=old_idx_name, new_name=new_idx_name)

    original_name = add_op.index.name
    add_op.reduce(rename_op, 'test_app')

    assert add_op.index.name == original_name, "reduce() should not mutate original operation"
```

**Failing input**: `model_name='A'`, `old_idx_name='A'`, `new_idx_name='H'`

## Reproducing the Bug

```python
from django.db.migrations.operations import AddIndex, RenameIndex
from django.db import models

index = models.Index(fields=['name'], name='old_idx')
add_op = AddIndex(model_name='TestModel', index=index)

print(f'Before reduce: {add_op.index.name}')

rename_op = RenameIndex(model_name='TestModel', old_name='old_idx', new_name='new_idx')
result = add_op.reduce(rename_op, 'test_app')

print(f'After reduce: {add_op.index.name}')
print(f'Expected: old_idx')
print(f'Actual: {add_op.index.name}')
```

## Why This Is A Bug

Looking at `django/db/migrations/operations/models.py` lines 992-994:

```python
if isinstance(operation, RenameIndex) and self.index.name == operation.old_name:
    self.index.name = operation.new_name
    return [self.__class__(model_name=self.model_name, index=self.index)]
```

The code directly mutates `self.index.name` before creating the new operation. This causes:

1. **Violation of Immutability**: The original `add_op` is modified, violating the documented immutability contract
2. **Shared State**: Both the original operation and the reduced operation share the same mutated index object
3. **Unexpected Behavior**: Code that retains references to the original operation will see its state change

## Fix

Create a new Index object instead of mutating the existing one:

```diff
diff --git a/django/db/migrations/operations/models.py b/django/db/migrations/operations/models.py
index abc123..def456 100644
--- a/django/db/migrations/operations/models.py
+++ b/django/db/migrations/operations/models.py
@@ -990,8 +990,12 @@ class AddIndex(IndexOperation):
         if isinstance(operation, RemoveIndex) and self.index.name == operation.name:
             return []
         if isinstance(operation, RenameIndex) and self.index.name == operation.old_name:
-            self.index.name = operation.new_name
-            return [self.__class__(model_name=self.model_name, index=self.index)]
+            # Create a new index with the new name instead of mutating the original
+            from django.db import models
+            new_index = models.Index(
+                fields=self.index.fields,
+                name=operation.new_name,
+            )
+            return [self.__class__(model_name=self.model_name, index=new_index)]
         return super().reduce(operation, app_label)
```