# Bug Report: attrs Hash/Equality Contract Violation

**Target**: `attrs.field` with `eq=False` and `hash=True`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `attrs` library allows defining fields with `eq=False` (excluded from equality) and `hash=True` (included in hash), which violates Python's fundamental hash/equality contract: equal objects must have equal hashes.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
import attrs

@given(st.integers(), st.integers(), st.integers())
def test_hash_equality_contract_with_eq_false_hash_true(shared, val1, val2):
    """Objects that are equal MUST have equal hashes (Python requirement)"""
    @attrs.define(hash=True)
    class Data:
        shared: int
        excluded: int = attrs.field(eq=False, hash=True)

    obj1 = Data(shared, val1)
    obj2 = Data(shared, val2)

    assume(val1 != val2)

    assert obj1 == obj2
    assert hash(obj1) == hash(obj2)
```

**Failing input**: `Data(0, 0)` and `Data(0, 1)`

## Reproducing the Bug

```python
import attrs

@attrs.define(hash=True)
class Data:
    shared: int
    excluded: int = attrs.field(eq=False, hash=True)

obj1 = Data(0, 1)
obj2 = Data(0, 2)

print(f"obj1 == obj2: {obj1 == obj2}")
print(f"hash(obj1): {hash(obj1)}")
print(f"hash(obj2): {hash(obj2)}")
print(f"Contract violated: {obj1 == obj2 and hash(obj1) != hash(obj2)}")
```

Output:
```
obj1 == obj2: True
hash(obj1): 5165053505784227494
hash(obj2): 271979476811531128
Contract violated: True
```

## Why This Is A Bug

According to Python's data model documentation:

> If a class defines `__eq__()`, it should also define `__hash__()` such that `a == b` implies `hash(a) == hash(b)`.

This is a fundamental Python invariant. When violated, it causes silent failures in hash-based data structures:

1. **Dictionary insertion fails silently**: Equal objects with different hashes can both exist as separate keys
2. **Set membership breaks**: Equal objects are treated as distinct set members
3. **Unpredictable behavior**: `obj in dict` may return False even when an equal object is a key

Example of the problem:
```python
obj1 = Data(0, 1)
obj2 = Data(0, 2)

d = {obj1: "first"}
d[obj2] = "second"

print(len(d))
print(obj1 == obj2)
```

This prints `2` and `True` - two equal objects are distinct dictionary keys!

## Fix

The attrs library should validate field parameters and raise an error when conflicting settings are used:

```diff
--- a/attr/_make.py
+++ b/attr/_make.py
@@ -2500,6 +2500,12 @@ def field(
         if eq is False and order is True:
             raise ValueError("eq must be True if order is True.")

+        # Validate hash/eq compatibility
+        if eq is False and hash is True:
+            raise ValueError(
+                "Cannot set hash=True when eq=False. This violates "
+                "Python's hash/equality contract: equal objects must have equal hashes."
+            )
+
         return _CountingAttr(
             default=default,
             validator=validator,
```

Alternatively, attrs could automatically set `hash=False` when `eq=False` is specified, or ignore the `hash=True` setting and issue a warning.