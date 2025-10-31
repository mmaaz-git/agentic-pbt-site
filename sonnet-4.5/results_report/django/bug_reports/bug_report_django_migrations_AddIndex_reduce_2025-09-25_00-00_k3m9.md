# Bug Report: django.db.migrations.operations.AddIndex violates immutability in reduce()

**Target**: `django.db.migrations.operations.AddIndex.reduce()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `AddIndex.reduce()` method mutates the operation's `index.name` attribute, violating the documented immutability guarantee of migration operations. This can lead to unexpected behavior when operations are reused or when the optimizer processes migration operations.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
from django.db.models import Index
from django.db.migrations.operations import AddIndex, RenameIndex

@given(
    model_name=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("Lu", "Ll"))),
    index_name=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("Lu", "Ll"))),
    new_index_name=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("Lu", "Ll"))),
)
@settings(max_examples=1000)
def test_add_index_immutability_in_reduce(model_name, index_name, new_index_name):
    index = Index(fields=["id"], name=index_name)
    op = AddIndex(model_name=model_name, index=index)

    original_index_name = op.index.name

    rename_op = RenameIndex(model_name=model_name, old_name=index_name, new_name=new_index_name)
    result = op.reduce(rename_op, "app")

    assert op.index.name == original_index_name
```

**Failing input**: `model_name='A', index_name='A', new_index_name='B'`

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

index = Index(fields=["id"], name="old_index")
add_op = AddIndex(model_name="MyModel", index=index)

print(f"Before reduce: {add_op.index.name}")

rename_op = RenameIndex(model_name="MyModel", old_name="old_index", new_name="new_index")
add_op.reduce(rename_op, "app")

print(f"After reduce: {add_op.index.name}")
```

## Why This Is A Bug

The `Operation` base class explicitly states at line 28 of `base.py`:

> Due to the way this class deals with deconstruction, it should be considered immutable.

However, `AddIndex.reduce()` at line 993 of `models.py` directly mutates the operation:

```python
if isinstance(operation, RenameIndex) and self.index.name == operation.old_name:
    self.index.name = operation.new_name  # MUTATION!
    return [self.__class__(model_name=self.model_name, index=self.index)]
```

This violates the immutability contract and can cause issues:
1. If the same operation instance is used multiple times, later uses will see the mutated state
2. The operation's behavior becomes order-dependent
3. Debugging becomes harder as operations change state unexpectedly

## Fix

```diff
--- a/django/db/migrations/operations/models.py
+++ b/django/db/migrations/operations/models.py
@@ -990,8 +990,10 @@ class AddIndex(IndexOperation):
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

Note: Django's `Index` class has a `clone()` method that creates a copy. If not, use `copy.deepcopy(self.index)` instead.