# Bug Report: Node.__copy__ Violates Shallow Copy Semantics

**Target**: `django.utils.tree.Node.__copy__`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `Node.__copy__()` method shares the children list reference between the original and copied nodes, violating Python's shallow copy semantics. This can cause unintended mutations to propagate between supposedly independent copies.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.utils import tree
from django.db.models.sql.where import AND, OR, XOR
import copy as python_copy


@given(st.sampled_from([AND, OR, XOR]), st.lists(st.integers(), min_size=1))
def test_copy_creates_independent_children_list(connector, children):
    node = tree.Node(children=children[:], connector=connector)
    copied = python_copy.copy(node)

    node.children.append(999)

    assert 999 not in copied.children
```

**Failing input**: `connector='AND', children=[1, 2, 3]`

## Reproducing the Bug

```python
import copy as python_copy
from django.utils import tree
from django.db.models.sql.where import AND

node = tree.Node(children=[1, 2, 3], connector=AND)
copied = python_copy.copy(node)

print(f"Children lists are same object: {copied.children is node.children}")

node.children.append(4)

print(f"Original children: {node.children}")
print(f"Copied children: {copied.children}")
print(f"Mutation affected copy: {4 in copied.children}")
```

Output:
```
Children lists are same object: True
Original children: [1, 2, 3, 4]
Copied children: [1, 2, 3, 4]
Mutation affected copy: True
```

## Why This Is A Bug

Python's copy protocol expects that `copy.copy()` creates a shallow copy where mutable container attributes are duplicated (but their contents may be shared). The current implementation shares the children list itself, meaning mutations to one node's children affect all copies. This violates the principle of least surprise and standard Python semantics.

While Django's internal usage in `Node.add()` works around this by immediately replacing `self.children`, external code or subclasses that use `copy()` and then mutate children will experience unexpected behavior.

## Fix

Change `__copy__` to create a new list for children:

```diff
--- a/django/utils/tree.py
+++ b/django/utils/tree.py
@@ -47,7 +47,7 @@ class Node:

     def __copy__(self):
         obj = self.create(connector=self.connector, negated=self.negated)
-        obj.children = self.children  # Don't [:] as .__init__() via .create() does.
+        obj.children = self.children[:]
         return obj
```