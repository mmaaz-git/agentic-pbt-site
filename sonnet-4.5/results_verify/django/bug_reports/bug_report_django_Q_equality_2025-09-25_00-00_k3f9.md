# Bug Report: django.db.models.Q Boolean Algebra Violations

**Target**: `django.db.models.Q`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Django's Q object equality comparison is order-dependent when combining Q objects with logical operators (`&`, `|`), violating fundamental boolean algebra properties including commutativity, idempotence, and De Morgan's laws.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from hypothesis import given, settings, strategies as st
from django.db.models import Q


@st.composite
def q_objects(draw):
    """Generate Q objects with simple field lookups."""
    field_names = ['id', 'name', 'value', 'count']
    field = draw(st.sampled_from(field_names))
    value = draw(st.one_of(
        st.integers(),
        st.text(min_size=0, max_size=10),
        st.booleans()
    ))
    return Q(**{field: value})


@given(q_objects(), q_objects())
@settings(max_examples=1000)
def test_q_and_commutative(q1, q2):
    """Q objects should satisfy commutativity for AND: q1 & q2 == q2 & q1."""
    result1 = q1 & q2
    result2 = q2 & q1
    assert result1 == result2, f"AND not commutative: {result1} != {result2}"
```

**Failing input**: `q1=Q(id=0)`, `q2=Q(id=1)`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.db.models import Q

q1 = Q(id=0)
q2 = Q(id=1)

result1 = q1 & q2
result2 = q2 & q1

print(f"Q(id=0) & Q(id=1) = {result1}")
print(f"Q(id=1) & Q(id=0) = {result2}")
print(f"Are they equal? {result1 == result2}")

print("\nExpected: True (AND is commutative)")
print(f"Actual: {result1 == result2}")

print("\n--- Idempotence Test ---")
q = Q(id=0)
result = q & q
print(f"Q(id=0) & Q(id=0) = {result}")
print(f"Q(id=0) = {q}")
print(f"Are they equal? {result == q}")
print(f"Expected: True (AND is idempotent)")
```

## Why This Is A Bug

The Q object class documentation states it can "combine filters logically (using `&` and `|`)". This implies that the logical operators follow boolean algebra rules. However, the current implementation violates fundamental boolean algebra properties:

1. **Commutativity**: `q1 & q2` should equal `q2 & q1`, but they don't
2. **Idempotence**: `q & q` should equal `q`, but they don't
3. **De Morgan's laws**: `~(q1 & q2)` should equal `~q1 | ~q2`, but they don't

The root cause is that when Q objects are combined using `&` or `|`, the children are preserved in the order they were combined. The `identity` property (used in `__eq__`) includes this ordering, making equality order-dependent.

In contrast, when Q objects are created with keyword arguments like `Q(id=0, name='a')`, Django correctly sorts the children (line 16 in `Q.__init__`: `children=[*args, *sorted(kwargs.items())]`), ensuring order-independent equality.

## Fix

The fix should ensure that Q object equality is order-independent for commutative operations (AND, OR). The `identity` property should sort children when the connector is AND or OR:

```diff
--- a/django/utils/tree.py
+++ b/django/utils/tree.py
@@ -50,7 +50,12 @@ class Node:
     @property
     def identity(self):
-        return self.__class__, *self.children
+        # For commutative operations (AND, OR), sort children for order-independent equality
+        if hasattr(self, 'connector') and self.connector in ('AND', 'OR'):
+            sorted_children = tuple(sorted(self.children, key=lambda x: (str(type(x)), str(x))))
+            return self.__class__, self.connector, self.negated, *sorted_children
+        else:
+            return self.__class__, *self.children
```

Alternatively, the fix could be in `Q._combine()` to sort children when creating the combined object:

```diff
--- a/django/db/models/query_utils.py
+++ b/django/db/models/query_utils.py
@@ -60,8 +60,11 @@ class Q(tree.Node):
         obj = self.create(connector=conn)
         obj.add(self, conn)
         obj.add(other, conn)
+        # Sort children for commutative operations to ensure order-independent equality
+        if conn in (self.AND, self.OR):
+            obj.children.sort(key=lambda x: (str(type(x)), str(x)))
         return obj
```