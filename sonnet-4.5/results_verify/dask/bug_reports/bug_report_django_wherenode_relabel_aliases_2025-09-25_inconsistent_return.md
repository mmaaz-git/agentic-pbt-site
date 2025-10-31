# Bug Report: WhereNode.relabel_aliases Inconsistent Return Value

**Target**: `django.db.models.sql.where.WhereNode.relabel_aliases`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `WhereNode.relabel_aliases()` method has inconsistent return behavior: it returns `self` when given an empty change_map, but returns `None` when given a non-empty change_map, violating API consistency.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from django.db.models.sql.where import WhereNode, AND, OR, XOR


@st.composite
def change_map_strategy(draw):
    num_mappings = draw(st.integers(min_value=1, max_value=5))
    change_map = {}
    for _ in range(num_mappings):
        old_alias = draw(st.text(min_size=1, max_size=10))
        new_alias = draw(st.text(min_size=1, max_size=10))
        change_map[old_alias] = new_alias
    return change_map


@given(st.sampled_from([AND, OR, XOR]), st.booleans(), change_map_strategy())
def test_wherenode_relabel_aliases_mutates_and_returns_self(connector, negated, change_map):
    node = WhereNode(connector=connector, negated=negated)
    result = node.relabel_aliases(change_map)
    assert result is node, "relabel_aliases should return self for method chaining"
```

**Failing input**: `connector='AND', negated=False, change_map={'old': 'new'}`

## Reproducing the Bug

```python
from django.db.models.sql.where import WhereNode, AND

node1 = WhereNode(connector=AND, negated=False)
result_empty = node1.relabel_aliases({})
print(f"Empty map returns: {result_empty}")
print(f"Returns self: {result_empty is node1}")

node2 = WhereNode(connector=AND, negated=False)
result_nonempty = node2.relabel_aliases({'old_alias': 'new_alias'})
print(f"Non-empty map returns: {result_nonempty}")
print(f"Returns self: {result_nonempty is node2}")
```

Output:
```
Empty map returns: <WhereNode: (AND: )>
Returns self: True
Non-empty map returns: None
Returns self: False
```

## Why This Is A Bug

This violates the principle of least surprise and breaks method chaining. The early return of `self` when the change_map is empty suggests the method is designed to support chaining, but the implicit `None` return in the non-empty case breaks this pattern. While Django's internal code doesn't currently chain this method, external code or future refactoring could reasonably expect consistent behavior.

## Fix

Add `return self` at the end of the `relabel_aliases` method:

```diff
--- a/django/db/models/sql/where.py
+++ b/django/db/models/sql/where.py
@@ -213,6 +213,7 @@ class WhereNode(tree.Node):
             child.relabel_aliases(change_map)
         elif hasattr(child, "relabeled_clone"):
             self.children[pos] = child.relabeled_clone(change_map)
+    return self
```