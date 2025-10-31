# Bug Report: attr Field Options Hash/Equality Contract Violation

**Target**: `attr.field` with `eq=False` and `hash=True`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When a field is configured with `eq=False` and `hash=True`, attrs creates classes that violate Python's fundamental hash/equality contract: if `a == b`, then `hash(a) == hash(b)`. This can cause silent data corruption when objects are used in sets or as dictionary keys.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import attr

@given(st.integers(), st.text(), st.text())
def test_eq_false_hash_true_contract_violation(val, text1, text2):
    from hypothesis import assume
    assume(text1 != text2)

    @attr.define(hash=True)
    class TestClass:
        x: int
        weird: str = attr.field(eq=False, hash=True)

    instance1 = TestClass(x=val, weird=text1)
    instance2 = TestClass(x=val, weird=text2)

    are_equal = (instance1 == instance2)
    if are_equal:
        assert hash(instance1) == hash(instance2), \
            "Hash/equality contract violated: equal objects have different hashes"
```

**Failing input**: `TestClass(x=1, weird='text1')` and `TestClass(x=1, weird='text2')`

## Reproducing the Bug

```python
import attr

@attr.define(hash=True)
class TestClass:
    x: int
    weird: str = attr.field(eq=False, hash=True)

instance1 = TestClass(x=1, weird="text1")
instance2 = TestClass(x=1, weird="text2")

print(f"instance1 == instance2: {instance1 == instance2}")
print(f"hash(instance1): {hash(instance1)}")
print(f"hash(instance2): {hash(instance2)}")
```

Output:
```
instance1 == instance2: True
hash(instance1): 4368964163948906462
hash(instance2): 6940874637871023909
```

## Why This Is A Bug

Python's data model requires that if two objects compare as equal, they must have the same hash value. From the Python documentation:

> If a class does not define an `__eq__()` method it should not define a `__hash__()` operation either; if it defines `__eq__()` but not `__hash__()`, its instances will not be usable as items in hashable collections. If a class defines mutable objects and implements an `__eq__()` method, it should not implement `__hash__()`, since the implementation of hashable collections requires that a key's hash value is immutable (if the object's hash value changes, it will be in the wrong hash bucket). **User-defined classes have `__eq__()` and `__hash__()` methods by default; with them, all objects compare unequal (except with themselves) and `x.__hash__()` returns an appropriate value such that `x == y` implies both that `x is y` and `hash(x) == hash(y)`.**

This bug causes:
- Silent failures when using these objects in sets (duplicate "equal" objects with different hashes)
- Dictionary corruption (equal keys stored in different buckets)
- Violation of mathematical invariants that code may depend on

## Fix

The attrs library should detect and prevent this configuration. When generating `__hash__`, attrs should ensure that all fields included in the hash are also included in equality comparison.

```diff
--- a/attr/_make.py
+++ b/attr/_make.py
@@ -somewhere in _make_hash
+    # Validate that hash/equality contract is maintained
+    eq_fields = {a.name for a in attrs if a.eq}
+    hash_fields = {a.name for a in attrs if a.hash is True or (a.hash is None and a.eq)}
+
+    # Fields in hash but not in eq violate the contract
+    problematic_fields = hash_fields - eq_fields
+    if problematic_fields:
+        raise ValueError(
+            f"Hash/equality contract violation: fields {problematic_fields} "
+            "are included in hash but excluded from equality. "
+            "If a == b, then hash(a) must equal hash(b). "
+            "Set hash=False on these fields or include them in equality."
+        )
```

Alternatively, attrs could document this as a dangerous configuration and automatically exclude fields with `eq=False` from the hash, or issue a warning.