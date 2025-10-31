# Bug Report: attrs.evolve Loses Field Values for init=False Fields

**Target**: `attrs.evolve`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `attrs.evolve()` function does not preserve the values of fields with `init=False`, instead resetting them to their default values. This is inconsistent with the similar function `attrs.assoc()`, which correctly preserves all field values.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import attrs


@attrs.define
class WithInitFalse:
    x: int
    y: int = attrs.field(init=False, default=0)


@given(st.integers(), st.integers())
def test_evolve_preserves_init_false_fields(x, y_value):
    instance = WithInitFalse(x=x)
    instance.y = y_value

    evolved = attrs.evolve(instance, x=x + 1)

    assert evolved.y == y_value
```

**Failing input**: Any instance where an `init=False` field has been modified from its default value.

## Reproducing the Bug

```python
import attrs


@attrs.define
class WithInitFalse:
    x: int
    y: int = attrs.field(init=False, default=0)


instance = WithInitFalse(x=5)
instance.y = 99

evolved = attrs.evolve(instance, x=10)
print(f"Original y: {instance.y}")
print(f"Evolved y: {evolved.y}")
```

Output:
```
Original y: 99
Evolved y: 0
```

## Why This Is A Bug

The `evolve()` function is documented to "Create a new instance, based on the first positional argument with *changes* applied." Users naturally expect this to mean creating a copy of the instance with only the specified changes, preserving all other field values.

However, `evolve()` internally calls `cls(**changes)` (line 616 in `attr/_make.py`), which reinitializes the instance. Fields with `init=False` are intentionally excluded from the `changes` dict (line 608-609), causing them to revert to their default values.

This behavior is particularly problematic because:

1. **Inconsistency with `assoc()`**: The deprecated-but-similar `assoc()` function uses `copy.copy()` and correctly preserves all field values, including `init=False` fields.

2. **Violates user expectations**: When a user calls `evolve(inst, x=new_x)`, they expect all fields except `x` to remain unchanged.

3. **Silent data loss**: The value loss happens silently without any warning or error.

## Fix

The fix should make `evolve()` preserve `init=False` field values. One approach:

```diff
--- a/attr/_make.py
+++ b/attr/_make.py
@@ -603,11 +603,17 @@ def evolve(*args, **changes):

     cls = inst.__class__
     attrs = fields(cls)
+    non_init_fields = {}
+
     for a in attrs:
         if not a.init:
-            continue
+            # Preserve the current value of init=False fields
+            non_init_fields[a.name] = getattr(inst, a.name)
+            continue
+
         attr_name = a.name  # To deal with private attributes.
         init_name = a.alias
         if init_name not in changes:
             changes[init_name] = getattr(inst, attr_name)

-    return cls(**changes)
+    new_inst = cls(**changes)
+    for name, value in non_init_fields.items():
+        _OBJ_SETATTR(new_inst, name, value)
+    return new_inst
```

This fix:
1. Collects values from `init=False` fields before creating the new instance
2. Creates the new instance with the `init=True` fields
3. Manually sets the `init=False` fields to their preserved values