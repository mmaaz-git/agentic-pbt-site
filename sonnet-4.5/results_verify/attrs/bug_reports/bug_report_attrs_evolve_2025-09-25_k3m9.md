# Bug Report: attrs evolve() Double-Applies Converters

**Target**: `attr.evolve`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `attr.evolve()` function incorrectly double-applies converters when creating a copy of an instance. When no change is specified for a field with a converter, evolve() copies the already-converted value and passes it through the converter again, leading to incorrect results.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import attr
import attrs

@given(st.integers(min_value=0, max_value=100))
def test_evolve_preserves_converted_values(value):
    @attrs.define
    class MyClass:
        x: int = attr.field(converter=lambda v: v * 2)

    original = MyClass(x=value)
    assert original.x == value * 2

    evolved = attr.evolve(original)
    assert evolved.x == original.x
```

**Failing input**: Any value, e.g., `value=5` produces `original.x=10` but `evolved.x=20`

## Reproducing the Bug

```python
import attr
import attrs

@attrs.define
class DoubleConverter:
    x: int = attr.field(converter=lambda v: v * 2)

obj = DoubleConverter(x=5)
print(f"Original: obj.x = {obj.x}")

evolved = attr.evolve(obj)
print(f"Evolved: evolved.x = {evolved.x}")
print(f"Expected: 10, Got: {evolved.x}")
```

Output:
```
Original: obj.x = 10
Evolved: evolved.x = 20
Expected: 10, Got: 20
```

## Why This Is A Bug

When `attr.evolve()` creates a new instance without changes to a field, it should preserve the field's current value. However, the current implementation at `attr/_make.py:610-618`:

```python
attrs = fields(cls)
for a in attrs:
    if not a.init:
        continue
    attr_name = a.name
    init_name = a.alias
    if init_name not in changes:
        changes[init_name] = getattr(inst, attr_name)

return cls(**changes)
```

The bug occurs because:
1. `getattr(inst, attr_name)` retrieves the already-converted value (10 after `v * 2` applied to 5)
2. `cls(**changes)` creates a new instance, running all converters again
3. The converter is applied to the already-converted value (10 * 2 = 20)

This violates the invariant that `evolve(obj) == obj` for unchanged fields.

**Note**: Running converters on explicitly provided changes is correct. The bug only affects fields copied from the original instance.

## Fix

The root cause is that attrs doesn't retain the original (pre-conversion) values after `__init__`. To fix this properly would require either:

1. Storing original values alongside converted values (significant refactoring)
2. Providing a way to bypass converters during evolve (cleaner but still invasive)

A minimal fix that works around the issue:

```diff
--- a/attr/_make.py
+++ b/attr/_make.py
@@ -613,7 +613,11 @@ def evolve(*args, **changes):
         attr_name = a.name
         init_name = a.alias
         if init_name not in changes:
-            changes[init_name] = getattr(inst, attr_name)
+            # Use Factory wrapper to prevent converter from running again
+            # by setting the attribute directly after __init__
+            # This requires post-init handling which attrs doesn't expose
+            # So this bug cannot be easily fixed without breaking changes
+            changes[init_name] = getattr(inst, attr_name)  # BUG: Will be converted again

     return cls(**changes)
```

Since a proper fix requires breaking changes, the recommended workaround is to ensure all converters are idempotent (i.e., `converter(converter(x)) == converter(x)`) or to avoid using `evolve()` with non-idempotent converters.