# Bug Report: Cython.Utility.Dataclasses Field repr Typo

**Target**: `Cython.Utility.Dataclasses.Field.__repr__`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `Field.__repr__` method uses `'kwonly='` in the format string, but the actual attribute name is `'kw_only'` (with underscore). This creates an inconsistency between the repr output and the actual attribute names.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from Cython.Utility.Dataclasses import field


@given(st.booleans() | st.none())
@settings(max_examples=100)
def test_field_repr_contains_kw_only(kw_only):
    result = field(kw_only=kw_only, default=42)
    repr_str = repr(result)

    assert "kw_only=" in repr_str, (
        f"Expected 'kw_only=' in repr output, but got:\n{repr_str}"
    )
```

**Failing input**: `kw_only=None` (or any value)

## Reproducing the Bug

```python
from Cython.Utility.Dataclasses import field

f = field(kw_only=True, default=42)
print(repr(f))
```

Output:
```
Field(name=None,type=None,default=42,default_factory=<...>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kwonly=True,)
```

Expected output should have `kw_only=True` instead of `kwonly=True`.

## Why This Is A Bug

The `Field` class defines the attribute as `kw_only` in its `__slots__` (line 32) and `__init__` (line 52), but the `__repr__` method (line 66) uses `kwonly` without the underscore. This inconsistency violates Python conventions where repr output should accurately reflect the object's attributes.

This makes debugging harder because users see `kwonly=` in the repr but must access the attribute as `field.kw_only`.

## Fix

Change line 66 in `Cython/Utility/Dataclasses.py` to use `kw_only=` instead of `kwonly=`:

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