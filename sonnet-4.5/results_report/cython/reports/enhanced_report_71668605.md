# Bug Report: Cython.Utility.Field.__repr__ Attribute Name Inconsistency

**Target**: `Cython.Utility.Dataclasses.Field.__repr__`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `Field.__repr__` method displays `kwonly=` in its output while the actual attribute is named `kw_only` (with underscore), creating an inconsistency between the repr output and the actual attribute name.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')
from Cython.Utility.Dataclasses import field

@given(st.booleans())
def test_field_repr_consistency(kw_only_value):
    f = field(kw_only=kw_only_value)
    repr_str = repr(f)
    # The repr should use the same attribute name as the actual attribute (kw_only, not kwonly)
    assert f'kw_only={kw_only_value!r}' in repr_str, \
        f"Expected 'kw_only={kw_only_value!r}' in repr, but got 'kwonly={kw_only_value!r}' instead. Full repr: {repr_str}"

if __name__ == "__main__":
    test_field_repr_consistency()
```

<details>

<summary>
**Failing input**: `kw_only_value=False`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 15, in <module>
    test_field_repr_consistency()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 7, in test_field_repr_consistency
    def test_field_repr_consistency(kw_only_value):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 11, in test_field_repr_consistency
    assert f'kw_only={kw_only_value!r}' in repr_str, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Expected 'kw_only=False' in repr, but got 'kwonly=False' instead. Full repr: Field(name=None,type=None,default=<Cython.Utility.Dataclasses._MISSING_TYPE object at 0x77bc3f7fd010>,default_factory=<Cython.Utility.Dataclasses._MISSING_TYPE object at 0x77bc3f7fd010>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kwonly=False,)
Falsifying example: test_field_repr_consistency(
    kw_only_value=False,
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Utility.Dataclasses import field

# Create a field with kw_only=True
f = field(kw_only=True)

# Print the repr to show the inconsistency
print("repr(f) output:")
print(repr(f))

print("\nAccessing the actual attribute:")
print(f"f.kw_only = {f.kw_only}")

print("\nTrying to access f.kwonly (without underscore):")
try:
    print(f"f.kwonly = {f.kwonly}")
except AttributeError as e:
    print(f"AttributeError: {e}")
```

<details>

<summary>
Demonstrates attribute naming inconsistency in repr output
</summary>
```
repr(f) output:
Field(name=None,type=None,default=<Cython.Utility.Dataclasses._MISSING_TYPE object at 0x78b9863ecd70>,default_factory=<Cython.Utility.Dataclasses._MISSING_TYPE object at 0x78b9863ecd70>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kwonly=True,)

Accessing the actual attribute:
f.kw_only = True

Trying to access f.kwonly (without underscore):
AttributeError: 'Field' object has no attribute 'kwonly'
```
</details>

## Why This Is A Bug

This violates the principle of consistency in API design. The Field class uses `kw_only` (with underscore) consistently throughout its implementation:
- The `__slots__` declaration includes `'kw_only'` (line 33)
- The constructor parameter is named `kw_only` (lines 38 and 102)
- The attribute is stored as `self.kw_only` (line 52)
- The field() function accepts `kw_only` parameter (line 102)
- But the `__repr__` format string uses `'kwonly={!r}'` without underscore (line 66)

This creates confusion when debugging, as the repr output displays an attribute name (`kwonly`) that doesn't actually exist on the object. The repr is meant to provide an accurate representation of an object's state, and using a different name than the actual attribute violates this expectation.

## Relevant Context

The Cython.Utility.Dataclasses module is a fallback implementation for when Python's standard library dataclasses module isn't available. As stated in the source comments, it "defines enough of the support types to be used with cdef classes."

The Python standard library's dataclasses module consistently uses `kw_only` with underscore throughout, making Cython's repr output inconsistent not only with itself but also with the standard library it's meant to emulate.

Source code location: `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Utility/Dataclasses.py`

## Proposed Fix

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