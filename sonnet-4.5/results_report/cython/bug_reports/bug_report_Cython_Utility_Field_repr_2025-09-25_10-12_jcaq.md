# Bug Report: Cython.Utility.Dataclasses.Field Inconsistent Attribute Name in __repr__

**Target**: `Cython.Utility.Dataclasses.Field.__repr__`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `Field.__repr__()` method displays the attribute as `kwonly=` in its string representation, but the actual attribute name is `kw_only` (with an underscore), creating an inconsistency between the repr and the actual object.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.Utility.Dataclasses import field, MISSING


@given(st.booleans())
def test_field_repr_uses_correct_attribute_name(kw_only_value):
    f = field(default=MISSING, kw_only=kw_only_value)
    repr_str = repr(f)

    assert f'kw_only={kw_only_value!r}' in repr_str, \
        f"Expected 'kw_only={kw_only_value!r}' in repr, but got: {repr_str}"
```

**Failing input**: `kw_only_value=False`

## Reproducing the Bug

```python
from Cython.Utility.Dataclasses import field, MISSING

f = field(default=MISSING, kw_only=True)

print("Field attributes:")
print(f"  f.kw_only = {f.kw_only}")

print("\nField repr:")
print(f"  {repr(f)}")
```

Output:
```
Field attributes:
  f.kw_only = True

Field repr:
  Field(name=None,type=None,default=<...>,default_factory=<...>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kwonly=True,)
```

## Why This Is A Bug

The `Field` class uses `kw_only` as its attribute name (defined in `__slots__` and used throughout the class), but the `__repr__` method incorrectly formats it as `kwonly=` without the underscore. This creates confusion when debugging or inspecting Field objects, as the repr suggests a different attribute name than what actually exists.

## Fix

```diff
--- a/Cython/Utility/Dataclasses.py
+++ b/Cython/Utility/Dataclasses.py
@@ -63,7 +63,7 @@ class Field:
                 'hash={!r},'
                 'compare={!r},'
                 'metadata={!r},'
-                'kwonly={!r},'
+                'kw_only={!r},'
                 ')'.format(self.name, self.type, self.default,
                            self.default_factory, self.init,
                            self.repr, self.hash, self.compare,
```