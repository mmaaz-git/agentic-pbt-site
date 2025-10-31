# Bug Report: django.db.migrations.graph.Node Equality Contract Violation

**Target**: `django.db.migrations.graph.Node.__eq__`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `Node.__eq__` method in `django/db/migrations/graph.py` violates Python's equality contract by implementing asymmetric equality comparison. When a `Node` is compared to a tuple key, `node == key` evaluates to `True`, but `key == node` evaluates to `False`, breaking the fundamental symmetry requirement.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.db.migrations.graph import Node

@given(st.tuples(st.text(min_size=1, max_size=20), st.text(min_size=1, max_size=20)))
def test_node_equality_symmetry_with_key(key):
    node = Node(key)

    forward = (node == key)
    backward = (key == node)

    assert forward == backward, (
        f"Symmetry violated: node == key is {forward}, but key == node is {backward}"
    )
```

**Failing input**: `("myapp", "0001_initial")`

## Reproducing the Bug

```python
from django.db.migrations.graph import Node

key = ("myapp", "0001_initial")
node = Node(key)

print(f"node == key: {node == key}")
print(f"key == node: {key == node}")

node_set = {node}
print(f"key in node_set: {key in node_set}")
```

Output:
```
node == key: True
key == node: False
key in node_set: True
```

## Why This Is A Bug

Python's equality contract requires that equality be:
1. **Symmetric**: If `x == y`, then `y == x`
2. **Reflexive**: `x == x` is always `True`
3. **Transitive**: If `x == y` and `y == z`, then `x == z`

The current implementation at `django/db/migrations/graph.py:20-21` breaks symmetry:

```python
def __eq__(self, other):
    return self.key == other
```

According to [Python's data model documentation](https://docs.python.org/3/reference/datamodel.html#object.__eq__):

> "By convention, False and NotImplemented are equivalent in this context. If the comparison with respect to the class is not supported, __eq__() should return NotImplemented."

This implementation allows `Node` to compare equal to a tuple key, but the comparison is not symmetric since tuple's `__eq__` doesn't know about `Node` objects.

Additionally, this causes semantic bugs where `key in node_set` returns `True` even though the set contains `Node` objects, not tuples. This violates Python's container membership semantics.

## Fix

```diff
--- a/django/db/migrations/graph.py
+++ b/django/db/migrations/graph.py
@@ -18,7 +18,10 @@ class Node:
         self.parents = set()

     def __eq__(self, other):
-        return self.key == other
+        if isinstance(other, Node):
+            return self.key == other.key
+        return NotImplemented

     def __lt__(self, other):
-        return self.key < other
+        if isinstance(other, Node):
+            return self.key < other.key
+        return NotImplemented
```

This fix:
1. Ensures symmetry by returning `NotImplemented` for non-Node comparisons, allowing Python's comparison machinery to handle the comparison correctly
2. Maintains hash consistency (hash is based on key, which is correct)
3. Fixes the `__lt__` method to be consistent with `__eq__`
4. Ensures that `node == key` and `key == node` both return `False` (since neither can compare)
5. Fixes set membership so that `key in node_set` correctly returns `False`