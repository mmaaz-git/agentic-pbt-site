# Bug Report: Cython.Utility.Dataclasses.Field Inconsistent Attribute Name in __repr__

**Target**: `Cython.Utility.Dataclasses.Field.__repr__`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `Field.__repr__()` method displays the attribute as `kwonly=` in its string representation, but the actual attribute name is `kw_only` (with an underscore), creating an inconsistency between the repr output and the actual object attribute.

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


if __name__ == "__main__":
    test_field_repr_uses_correct_attribute_name()
```

<details>

<summary>
**Failing input**: `kw_only_value=False`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/41/hypo.py", line 15, in <module>
    test_field_repr_uses_correct_attribute_name()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/41/hypo.py", line 6, in test_field_repr_uses_correct_attribute_name
    def test_field_repr_uses_correct_attribute_name(kw_only_value):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/41/hypo.py", line 10, in test_field_repr_uses_correct_attribute_name
    assert f'kw_only={kw_only_value!r}' in repr_str, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Expected 'kw_only=False' in repr, but got: Field(name=None,type=None,default=<Cython.Utility.Dataclasses._MISSING_TYPE object at 0x76e2d8a25160>,default_factory=<Cython.Utility.Dataclasses._MISSING_TYPE object at 0x76e2d8a25160>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kwonly=False,)
Falsifying example: test_field_repr_uses_correct_attribute_name(
    kw_only_value=False,
)
```
</details>

## Reproducing the Bug

```python
from Cython.Utility.Dataclasses import field, MISSING

f = field(default=MISSING, kw_only=True)

print("Field attributes:")
print(f"  f.kw_only = {f.kw_only}")

print("\nField repr:")
print(f"  {repr(f)}")

print("\nChecking repr content:")
if 'kw_only=True' in repr(f):
    print("  ✓ repr contains 'kw_only=True'")
else:
    print("  ✗ repr does NOT contain 'kw_only=True'")

if 'kwonly=True' in repr(f):
    print("  ✓ repr contains 'kwonly=True'")
else:
    print("  ✗ repr does NOT contain 'kwonly=True'")
```

<details>

<summary>
Field repr shows 'kwonly=True' instead of 'kw_only=True'
</summary>
```
Field attributes:
  f.kw_only = True

Field repr:
  Field(name=None,type=None,default=<Cython.Utility.Dataclasses._MISSING_TYPE object at 0x7472253f0d70>,default_factory=<Cython.Utility.Dataclasses._MISSING_TYPE object at 0x7472253f0d70>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kwonly=True,)

Checking repr content:
  ✗ repr does NOT contain 'kw_only=True'
  ✓ repr contains 'kwonly=True'
```
</details>

## Why This Is A Bug

This violates the expected behavior of a `__repr__` method which should accurately represent the object's actual attributes. The `Field` class consistently uses `kw_only` (with underscore) as the attribute name throughout its implementation:
- Line 33 of `Dataclasses.py`: `__slots__` defines the attribute as `'kw_only'`
- Line 52: The constructor sets `self.kw_only = kw_only`
- Line 102: The `field()` function accepts a `kw_only` parameter
- Line 70: The `__repr__` method accesses `self.kw_only` but formats it as `'kwonly={!r}'` on line 66

This creates confusion when debugging or inspecting Field objects, as the repr suggests accessing an attribute named `kwonly` (without underscore) which doesn't exist. Attempting to access `f.kwonly` would raise an AttributeError, while the actual attribute `f.kw_only` works fine.

Additionally, this is a fallback implementation meant to provide compatibility when Python's standard `dataclasses` module is unavailable. Python's standard `dataclasses.Field` uses `kw_only` with an underscore in both the attribute name and its repr, so this inconsistency breaks compatibility expectations.

## Relevant Context

This bug exists in Cython's fallback dataclasses implementation, which is used when the standard library's dataclasses module isn't available. The module is located at `/Cython/Utility/Dataclasses.py` and includes a comment at the top stating it's "the fallback dataclass code if the stdlib module isn't available."

The Python standard library's `dataclasses.Field` class (which this code is meant to mimic) consistently uses `kw_only` with an underscore in both the attribute name and repr output. You can verify this in Python 3.10+ with:
```python
import dataclasses
f = dataclasses.Field(default=dataclasses.MISSING, kw_only=True, ...)
print(repr(f))  # Shows 'kw_only=True', not 'kwonly=True'
```

Source code location: https://github.com/cython/cython/blob/master/Cython/Utility/Dataclasses.py

## Proposed Fix

The fix is straightforward - change line 66 in `Dataclasses.py` to use the correct attribute name in the repr format string:

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