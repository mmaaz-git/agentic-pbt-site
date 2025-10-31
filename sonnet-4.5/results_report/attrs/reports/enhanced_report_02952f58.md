# Bug Report: attrs evolve() Double-Applies Converters to Unchanged Fields

**Target**: `attr.evolve`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `attr.evolve()` function incorrectly applies converters twice to unchanged fields when creating a copy of an instance, resulting in silent data corruption for non-idempotent converters.

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

if __name__ == "__main__":
    test_evolve_preserves_converted_values()
```

<details>

<summary>
**Failing input**: `value=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/55/hypo.py", line 18, in <module>
    test_evolve_preserves_converted_values()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/55/hypo.py", line 6, in test_evolve_preserves_converted_values
    def test_evolve_preserves_converted_values(value):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/55/hypo.py", line 15, in test_evolve_preserves_converted_values
    assert evolved.x == original.x
           ^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_evolve_preserves_converted_values(
    value=1,
)
```
</details>

## Reproducing the Bug

```python
import attr
import attrs

@attrs.define
class DoubleConverter:
    x: int = attr.field(converter=lambda v: v * 2)

# Create an instance with x=5
# The converter will make x = 5 * 2 = 10
obj = DoubleConverter(x=5)
print(f"Original: obj.x = {obj.x}")

# Evolve without changing anything
# This should preserve x=10, but it applies the converter again
evolved = attr.evolve(obj)
print(f"Evolved: evolved.x = {evolved.x}")
print(f"Expected: 10, Got: {evolved.x}")

# Verify the bug happens every time
print("\nDemonstrating the issue with different values:")
for value in [1, 3, 7, 10]:
    obj = DoubleConverter(x=value)
    evolved = attr.evolve(obj)
    print(f"  Value={value}: original.x={obj.x}, evolved.x={evolved.x}, should be equal but aren't")
```

<details>

<summary>
Double-application of converter on unchanged fields during evolve()
</summary>
```
Original: obj.x = 10
Evolved: evolved.x = 20
Expected: 10, Got: 20

Demonstrating the issue with different values:
  Value=1: original.x=2, evolved.x=4, should be equal but aren't
  Value=3: original.x=6, evolved.x=12, should be equal but aren't
  Value=7: original.x=14, evolved.x=28, should be equal but aren't
  Value=10: original.x=20, evolved.x=40, should be equal but aren't
```
</details>

## Why This Is A Bug

This behavior violates the fundamental expectation that `evolve(obj)` without any changes should produce an identical copy of the object. The current implementation has a critical flaw in how it handles converters:

1. **Converters are meant to transform input values during initialization**: When you create `DoubleConverter(x=5)`, the converter transforms the input value 5 into 10, which is stored as the attribute value.

2. **evolve() incorrectly re-applies converters to already-converted values**: When `evolve()` copies unchanged fields, it retrieves the already-converted value (10) and passes it through `__init__`, which applies the converter again (10 → 20).

3. **The documentation does not specify this behavior**: The attrs documentation for `evolve()` states it "creates a new instance based on an existing instance with specified changes" but makes no mention that converters will be re-applied to unchanged fields.

4. **This contradicts the principle of least surprise**: Users reasonably expect that:
   - `evolve(obj) == obj` when no changes are specified
   - Converters only apply to new input values, not already-converted stored values
   - Unchanged fields remain unchanged

5. **Silent data corruption occurs**: Non-idempotent converters (like multipliers, counters, timestamp generators, unique ID generators) will produce incorrect results without any warning or error.

## Relevant Context

The bug occurs in the `evolve()` implementation at `/lib/python3.13/site-packages/attr/_make.py` lines 610-618:

```python
attrs = fields(cls)
for a in attrs:
    if not a.init:
        continue
    attr_name = a.name  # To deal with private attributes.
    init_name = a.alias
    if init_name not in changes:
        changes[init_name] = getattr(inst, attr_name)  # Gets converted value

return cls(**changes)  # Re-runs converters on all values
```

The issue is that `getattr(inst, attr_name)` retrieves the already-converted value from the instance, but `cls(**changes)` runs `__init__` which applies all converters again.

**Common use cases affected by this bug:**
- Counter/incrementor converters: `converter=lambda x: x + counter.next()`
- Timestamp converters: `converter=lambda x: f"{x}_{datetime.now()}"`
- Transformation chains: `converter=lambda x: transform(normalize(x))`
- Unique ID generators: `converter=lambda x: f"{x}_{uuid.uuid4()}"`

**Workaround**: Users must ensure all converters are idempotent (i.e., `f(f(x)) == f(x)`), which significantly limits converter functionality.

**Related documentation:**
- attrs evolve documentation: https://www.attrs.org/en/stable/api.html#attr.evolve
- attrs converters documentation: https://www.attrs.org/en/stable/init.html#converters

This bug affects attrs version 25.3.0 and likely earlier versions.

## Proposed Fix

The fundamental issue is that attrs doesn't retain the original pre-conversion values after initialization, making it impossible to distinguish between "value that should be converted" and "value that was already converted". A proper fix would require significant architectural changes. However, here's a minimal fix that could work by detecting when we're in an evolve context:

```diff
--- a/attr/_make.py
+++ b/attr/_make.py
@@ -608,11 +608,20 @@ def evolve(*args, **changes):

     cls = inst.__class__
     attrs = fields(cls)
+    unchanged_attrs = set()
     for a in attrs:
         if not a.init:
             continue
         attr_name = a.name  # To deal with private attributes.
         init_name = a.alias
         if init_name not in changes:
+            unchanged_attrs.add(attr_name)
             changes[init_name] = getattr(inst, attr_name)

-    return cls(**changes)
+    # Create instance with special marker to skip converters for unchanged fields
+    # This would require modifying __init__ generation to check for this marker
+    with _evolve_context(unchanged_attrs):
+        return cls(**changes)
```

Since this requires invasive changes to the initialization code generation, a more practical short-term fix would be to clearly document this limitation and recommend idempotent converters when using `evolve()`.