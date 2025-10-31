# Bug Report: django.db.models.Q Equality Violates Boolean Algebra Properties

**Target**: `django.db.models.Q`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The Q class implements `__eq__` and `__hash__` using structural comparison rather than logical equivalence, causing Q objects with identical logical meaning but different structure to be treated as unequal. This violates fundamental Boolean algebra properties (commutativity and idempotence) that users would reasonably expect from logical AND/OR operators.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from hypothesis import given, strategies as st, settings, assume
from django.db.models import Q


@st.composite
def q_objects(draw):
    field_name = draw(st.sampled_from(['name', 'age', 'city', 'email', 'status']))
    value = draw(st.one_of(
        st.text(min_size=1, max_size=20),
        st.integers(min_value=0, max_value=100),
        st.booleans()
    ))
    return Q(**{field_name: value})


@given(q_objects(), q_objects())
@settings(max_examples=200)
def test_q_and_commutativity(q1, q2):
    assume(q1 != q2)
    result1 = q1 & q2
    result2 = q2 & q1
    assert result1 == result2, f"Expected {result1} == {result2}"


@given(q_objects())
@settings(max_examples=200)
def test_q_and_idempotence(q):
    result = q & q
    assert result == q, f"Expected {result} == {q}"
```

**Failing input**: `q1=Q(name='Alice'), q2=Q(age=30)` for commutativity; `q=Q(name='Alice')` for idempotence

## Reproducing the Bug

```python
from django.db.models import Q

q1 = Q(name='Alice')
q2 = Q(age=30)

and_12 = q1 & q2
and_21 = q2 & q1

print(f"q1 & q2 = {and_12}")
print(f"q2 & q1 = {and_21}")
print(f"Equal? {and_12 == and_21}")
print(f"Same hash? {hash(and_12) == hash(and_21)}")

filters = {and_12, and_21}
print(f"Set contains {len(filters)} items (expected 1)")

q = Q(name='Alice')
q_dup = q & q
print(f"q = {q}")
print(f"q & q = {q_dup}")
print(f"Equal? {q == q_dup}")
```

Output:
```
q1 & q2 = (AND: ('name', 'Alice'), ('age', 30))
q2 & q1 = (AND: ('age', 30), ('name', 'Alice'))
Equal? False
Same hash? False
Set contains 2 items (expected 1)
q = (AND: ('name', 'Alice'))
q & q = (AND: ('name', 'Alice'), ('name', 'Alice'))
Equal? False
```

## Why This Is A Bug

1. **Violates commutativity**: In Boolean algebra, `a AND b = b AND a`. Users would expect `q1 & q2 == q2 & q1` since the logical meaning is identical.

2. **Violates idempotence**: In Boolean algebra, `a AND a = a`. The expression `q & q` should equal `q`, but instead creates a duplicate condition.

3. **Breaks set deduplication**: When Q objects are added to a set for deduplication, logically identical queries are treated as different due to structural differences.

4. **Breaks dict-based caching**: Using Q objects as dictionary keys would create redundant cache entries for logically equivalent queries.

5. **Violates API contract**: The Q class implements `__hash__` and `__eq__`, suggesting it's designed for use in sets and as dict keys. The docstring says Q objects can be "combined logically", implying logical equivalence should be respected.

## Fix

The `identity` property in `django/db/models/query_utils.py` should normalize the children order to ensure logically equivalent Q objects are equal:

```diff
--- a/django/db/models/query_utils.py
+++ b/django/db/models/query_utils.py
@@ -204,7 +204,7 @@ class Q(tree.Node):
     @cached_property
     def identity(self):
         path, args, kwargs = self.deconstruct()
-        identity = [path, *kwargs.items()]
+        identity = [path, *sorted(kwargs.items())]
         for child in args:
             if isinstance(child, tuple):
                 arg, value = child
@@ -212,7 +212,7 @@ class Q(tree.Node):
                 identity.append((arg, value))
             else:
                 identity.append(child)
-        return tuple(identity)
+        return tuple([identity[0]] + sorted(identity[1:], key=lambda x: (str(type(x)), str(x))))
```

Note: A more comprehensive fix would also handle idempotence by normalizing duplicate children, but that requires more careful consideration of the tree structure. The above fix addresses commutativity at minimum.