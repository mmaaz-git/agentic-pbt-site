# Bug Report: django.utils.tree.Node.__copy__ Shares Mutable Children List Reference

**Target**: `django.utils.tree.Node.__copy__`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `Node.__copy__()` method in Django's tree utility module shares the same list object between the original and copied nodes, violating Python's shallow copy conventions and causing mutations to propagate between supposedly independent copies.

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

# Run the test
if __name__ == "__main__":
    test_copy_creates_independent_children_list()
```

<details>

<summary>
**Failing input**: `connector='AND', children=[0]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 18, in <module>
    test_copy_creates_independent_children_list()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 8, in test_copy_creates_independent_children_list
    def test_copy_creates_independent_children_list(connector, children):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 14, in test_copy_creates_independent_children_list
    assert 999 not in copied.children
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_copy_creates_independent_children_list(
    # The test always failed when commented parts were varied together.
    connector='AND',  # or any other generated value
    children=[0],  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import copy as python_copy
from django.utils import tree
from django.db.models.sql.where import AND

# Create a Node with children list containing integers
node = tree.Node(children=[1, 2, 3], connector=AND)

# Make a shallow copy using Python's copy module
copied = python_copy.copy(node)

# Check if the children lists are the same object (they should not be for proper shallow copy)
print(f"Children lists are same object: {copied.children is node.children}")
print(f"Original children id: {id(node.children)}")
print(f"Copied children id: {id(copied.children)}")
print()

# Modify the original node's children list
print("Before mutation:")
print(f"Original children: {node.children}")
print(f"Copied children: {copied.children}")
print()

node.children.append(4)

print("After appending 4 to original node's children:")
print(f"Original children: {node.children}")
print(f"Copied children: {copied.children}")
print(f"Mutation affected copy: {4 in copied.children}")
```

<details>

<summary>
Output demonstrates shared reference causing unintended mutation propagation
</summary>
```
Children lists are same object: True
Original children id: 124899307637568
Copied children id: 124899307637568

Before mutation:
Original children: [1, 2, 3]
Copied children: [1, 2, 3]

After appending 4 to original node's children:
Original children: [1, 2, 3, 4]
Copied children: [1, 2, 3, 4]
Mutation affected copy: True
```
</details>

## Why This Is A Bug

This violates Python's established shallow copy semantics where mutable container attributes should be duplicated, not shared. The Node class implements container-like behavior through `__len__`, `__contains__`, and `__bool__` methods that all operate on the `children` attribute, making it a container type that should follow Python's conventions.

All Python standard library container types (list, dict, deque, OrderedDict, etc.) create new container objects during shallow copy while sharing the contained elements. The current implementation in Django violates this fundamental convention by sharing the container itself (the children list), not just its elements.

The code at line 49 of `/home/npc/miniconda/lib/python3.13/site-packages/django/utils/tree.py` explicitly shares the reference with a comment "Don't [:] as .__init__() via .create() does." However, this intentional behavior contradicts Python's copy protocol and causes unexpected mutations between copies, violating the principle of least surprise.

## Relevant Context

The Node class is located in `django/utils/tree.py` and is primarily used for filter constructs in Django's ORM. The class is subclassed by `WhereNode` in `django/db/models/sql/where.py` for SQL WHERE clause generation.

While Django's internal usage in the `Node.add()` method (lines 102-104) works correctly by immediately replacing `self.children` after copying, any external code or Django extensions that use `copy.copy()` on Node instances and then mutate the children list will experience unexpected shared mutations.

The `__deepcopy__` method (lines 54-57) correctly creates a new list with `copy.deepcopy(self.children, memodict)`, showing that deep copy follows proper semantics while shallow copy does not.

Documentation: https://docs.djangoproject.com/en/stable/ref/models/querysets/
Source code: https://github.com/django/django/blob/main/django/utils/tree.py

## Proposed Fix

```diff
--- a/django/utils/tree.py
+++ b/django/utils/tree.py
@@ -46,7 +46,7 @@ class Node:

     def __copy__(self):
         obj = self.create(connector=self.connector, negated=self.negated)
-        obj.children = self.children  # Don't [:] as .__init__() via .create() does.
+        obj.children = self.children[:]
         return obj
```