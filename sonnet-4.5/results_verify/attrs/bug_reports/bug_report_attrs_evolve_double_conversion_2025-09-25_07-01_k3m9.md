# Bug Report: attrs.evolve Double-Conversion

**Target**: `attrs.evolve`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`attrs.evolve()` applies converters to unchanged fields, causing double-conversion when creating a copy with partial changes. This silently corrupts field values that were not meant to be modified.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import attrs

@given(st.integers(), st.text())
def test_evolve_does_not_double_convert_unchanged_fields(initial_value, new_string):
    conversion_count = 0

    def tracking_converter(value):
        nonlocal conversion_count
        conversion_count += 1
        return value * 2

    @attrs.define
    class TestClass:
        x: int = attrs.field(converter=tracking_converter)
        y: str = "default"

    conversion_count = 0
    obj1 = TestClass(initial_value, "original")
    assert conversion_count == 1
    assert obj1.x == initial_value * 2

    conversion_count = 0
    obj2 = attrs.evolve(obj1, y=new_string)

    assert conversion_count == 0, \
        f"Converter called {conversion_count} times during evolve"
    assert obj2.x == obj1.x, \
        f"Unchanged field modified: {obj1.x} -> {obj2.x}"
```

**Failing input**: Any input where a field has a converter and is not explicitly changed in the evolve call.

## Reproducing the Bug

```python
import attrs

def double_it(x):
    return x * 2

@attrs.define
class Container:
    x: int = attrs.field(converter=double_it)
    y: str = "default"

obj1 = Container(5, "hello")
print(f"obj1.x = {obj1.x}")

obj2 = attrs.evolve(obj1, y="world")
print(f"obj2.x = {obj2.x}")

assert obj1.x == 10
assert obj2.x == 20
```

## Why This Is A Bug

When `evolve(obj, y="world")` is called:
1. The evolve function retrieves the current value of `x` from `obj1` via `getattr(obj1, "x")`, which returns 10 (the already-converted value)
2. It then creates a new instance with `Container(x=10, y="world")`
3. The `__init__` method runs the converter on `x=10`, converting it to 20
4. The result is that `obj2.x = 20` instead of the expected `obj2.x = 10`

This violates the fundamental expectation that `evolve` creates a copy with only the specified changes applied. Fields not mentioned in the changes should remain identical to the original instance.

This causes silent data corruption - users won't get an error, just wrong values. It's especially problematic with:
- Converters that are not idempotent
- Converters that have side effects
- Converters that transform data (e.g., `str.lower`, normalization functions)

## Fix

The issue is in `/home/npc/pbt/agentic-pbt/envs/attrs_env/lib/python3.13/site-packages/attr/_make.py` lines 609-618:

```python
def evolve(*args, **changes):
    # ...
    cls = inst.__class__
    attrs = fields(cls)
    for a in attrs:
        if not a.init:
            continue
        attr_name = a.name
        init_name = a.alias
        if init_name not in changes:
            changes[init_name] = getattr(inst, attr_name)  # Gets converted value

    return cls(**changes)  # Re-runs converters!
```

The fix requires bypassing converters for unchanged fields. One approach:

```diff
--- a/attr/_make.py
+++ b/attr/_make.py
@@ -608,11 +608,23 @@ def evolve(*args, **changes):
     cls = inst.__class__
     attrs = fields(cls)
+
+    # Collect fields that should bypass conversion
+    unchanged_fields = {}
     for a in attrs:
         if not a.init:
             continue
         attr_name = a.name
         init_name = a.alias
         if init_name not in changes:
-            changes[init_name] = getattr(inst, attr_name)
+            if a.converter is not None:
+                # Field has a converter and is unchanged - bypass it
+                unchanged_fields[attr_name] = getattr(inst, attr_name)
+            else:
+                changes[init_name] = getattr(inst, attr_name)

-    return cls(**changes)
+    # Create new instance
+    new_inst = cls(**changes)
+    # Set unchanged converted fields directly to bypass converters
+    for attr_name, value in unchanged_fields.items():
+        _OBJ_SETATTR(new_inst, attr_name, value)
+    return new_inst
```

This fix creates the new instance with only the truly changed fields, then directly sets the unchanged fields to bypass converters.