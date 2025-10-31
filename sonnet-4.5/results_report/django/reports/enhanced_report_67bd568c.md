# Bug Report: django.db.migrations.operations.AddIndex.reduce Mutates Original Index Object

**Target**: `django.db.migrations.operations.AddIndex.reduce`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `AddIndex.reduce()` method mutates the original operation's index object when reducing with a `RenameIndex` operation, violating the immutability pattern followed by all other reduce() implementations in Django's migration operations.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Property-based test using Hypothesis to verify AddIndex.reduce() immutability
"""
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

    assert add_op.index.name == original_name, f"reduce() should not mutate original operation: expected {original_name}, got {add_op.index.name}"

# Run the test
if __name__ == "__main__":
    test_add_index_reduce_immutability()
```

<details>

<summary>
**Failing input**: `model_name='A', old_idx_name='A', new_idx_name='B'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/46/hypo.py", line 38, in <module>
    test_add_index_reduce_immutability()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/46/hypo.py", line 24, in test_add_index_reduce_immutability
    def test_add_index_reduce_immutability(model_name, old_idx_name, new_idx_name):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/46/hypo.py", line 34, in test_add_index_reduce_immutability
    assert add_op.index.name == original_name, f"reduce() should not mutate original operation: expected {original_name}, got {add_op.index.name}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: reduce() should not mutate original operation: expected A, got B
Falsifying example: test_add_index_reduce_immutability(
    model_name='A',
    old_idx_name='A',
    new_idx_name='B',
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Minimal reproduction case for AddIndex.reduce() mutation bug in Django
"""
from django.db.migrations.operations import AddIndex, RenameIndex
from django.db import models

# Create an index with an initial name
index = models.Index(fields=['name'], name='old_idx')

# Create an AddIndex operation
add_op = AddIndex(model_name='TestModel', index=index)

# Print the initial state
print(f'Before reduce: {add_op.index.name}')

# Create a RenameIndex operation
rename_op = RenameIndex(model_name='TestModel', old_name='old_idx', new_name='new_idx')

# Call reduce() which should NOT mutate the original operation
result = add_op.reduce(rename_op, 'test_app')

# Check the state after reduce
print(f'After reduce: {add_op.index.name}')
print(f'Expected: old_idx')
print(f'Actual: {add_op.index.name}')

# Show that the mutation happened
if add_op.index.name != 'old_idx':
    print('\n❌ BUG CONFIRMED: The original AddIndex operation was mutated!')
    print('   The index.name changed from "old_idx" to "new_idx"')
else:
    print('\n✓ No bug: The original operation was not mutated')
```

<details>

<summary>
Bug Confirmed: Original AddIndex operation was mutated
</summary>
```
Before reduce: old_idx
After reduce: new_idx
Expected: old_idx
Actual: new_idx

❌ BUG CONFIRMED: The original AddIndex operation was mutated!
   The index.name changed from "old_idx" to "new_idx"
```
</details>

## Why This Is A Bug

This violates expected behavior for several critical reasons:

1. **Breaks Established Pattern**: Every other reduce() method in Django's migration operations creates new objects instead of mutating the original. For example, `RenameIndex.reduce()`, `AddConstraint.reduce()`, `CreateModel.reduce()`, `AddField.reduce()`, etc., all follow the pattern of creating new instances rather than modifying existing ones.

2. **Violates Immutability Contract**: While not explicitly documented, the consistent implementation pattern across all other operations establishes an implicit contract that reduce() methods should not mutate the original operation. Code that retains references to the original operation will unexpectedly see its state change.

3. **Shared State Problem**: The bug occurs at line 993 in `/home/npc/miniconda/lib/python3.13/site-packages/django/db/migrations/operations/models.py`:
   ```python
   if isinstance(operation, RenameIndex) and self.index.name == operation.old_name:
       self.index.name = operation.new_name  # <-- MUTATES the original!
       return [self.__class__(model_name=self.model_name, index=self.index)]
   ```
   Both the original `add_op` and the returned reduced operation share the same index object, which has been mutated.

4. **Potential for Subtle Bugs**: Migration optimizers or analysis tools that work with migration operations may rely on operations being immutable during reduction. This mutation could lead to hard-to-debug issues in migration squashing, optimization, or custom migration tooling.

## Relevant Context

The issue is specifically in the `AddIndex.reduce()` method in Django's migration operations framework. This method is called during migration optimization to simplify sequences of operations.

The Django migration system documentation doesn't explicitly state that operations must be immutable, but the consistent pattern across all other operations (verified by examining the source code) makes this deviation a clear bug rather than a design choice.

Django version tested: 5.2.6

Relevant Django source file: `django/db/migrations/operations/models.py`

## Proposed Fix

Create a new Index object with the updated name instead of mutating the existing one:

```diff
--- a/django/db/migrations/operations/models.py
+++ b/django/db/migrations/operations/models.py
@@ -990,8 +990,17 @@ class AddIndex(IndexOperation):
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
+                condition=getattr(self.index, 'condition', None),
+                include=getattr(self.index, 'include', None),
+                opclasses=getattr(self.index, 'opclasses', None),
+                expressions=getattr(self.index, 'expressions', None),
+                db_tablespace=getattr(self.index, 'db_tablespace', None),
+            )
+            return [self.__class__(model_name=self.model_name, index=new_index)]
         return super().reduce(operation, app_label)
```