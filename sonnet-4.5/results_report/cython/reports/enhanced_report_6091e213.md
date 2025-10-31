# Bug Report: Cython.Utility.Dataclasses Field.__repr__ Attribute Name Mismatch

**Target**: `Cython.Utility.Dataclasses.Field.__repr__`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `Field.__repr__` method displays `kwonly=` in its output string, but the actual attribute name is `kw_only` (with underscore), causing an inconsistency between repr output and the object's actual attributes.

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

if __name__ == "__main__":
    # Run the test and catch the first failure
    test_field_repr_contains_kw_only()
```

<details>

<summary>
**Failing input**: `kw_only=None`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/38/hypo.py", line 17, in <module>
    test_field_repr_contains_kw_only()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/38/hypo.py", line 6, in test_field_repr_contains_kw_only
    @settings(max_examples=100)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/38/hypo.py", line 11, in test_field_repr_contains_kw_only
    assert "kw_only=" in repr_str, (
           ^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Expected 'kw_only=' in repr output, but got:
Field(name=None,type=None,default=42,default_factory=<Cython.Utility.Dataclasses._MISSING_TYPE object at 0x75a32468d010>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kwonly=None,)
Falsifying example: test_field_repr_contains_kw_only(
    kw_only=None,
)
```
</details>

## Reproducing the Bug

```python
from Cython.Utility.Dataclasses import field

# Create a field with kw_only=True and default=42
f = field(kw_only=True, default=42)

# Print the repr output to show the issue
print("repr(f) output:")
print(repr(f))

# Show that the actual attribute is 'kw_only' not 'kwonly'
print("\nAccessing f.kw_only:")
print(f"f.kw_only = {f.kw_only}")

print("\nTrying to access f.kwonly (will raise AttributeError):")
try:
    print(f.kwonly)
except AttributeError as e:
    print(f"AttributeError: {e}")
```

<details>

<summary>
repr() displays "kwonly=" but attribute is actually "kw_only"
</summary>
```
repr(f) output:
Field(name=None,type=None,default=42,default_factory=<Cython.Utility.Dataclasses._MISSING_TYPE object at 0x7fb6b47f0d70>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kwonly=True,)

Accessing f.kw_only:
f.kw_only = True

Trying to access f.kwonly (will raise AttributeError):
AttributeError: 'Field' object has no attribute 'kwonly'
```
</details>

## Why This Is A Bug

This violates the Python convention that `__repr__` output should accurately reflect an object's actual attributes. The inconsistency creates confusion during debugging:

1. The Field class defines the attribute as `kw_only` in its `__slots__` (line 33 of Dataclasses.py)
2. The `__init__` method correctly sets `self.kw_only = kw_only` (line 52)
3. The `__repr__` method incorrectly formats the output with `'kwonly={!r}'` instead of `'kw_only={!r}'` (line 66)
4. The format arguments correctly reference `self.kw_only` (line 70), but the template string has the typo

This breaks compatibility with Python's standard library dataclasses module, where the Field repr correctly displays `kw_only=`. Users seeing `kwonly=` in debug output may attempt to access `field.kwonly`, which will fail with AttributeError since the actual attribute is `field.kw_only`.

## Relevant Context

The bug is located in `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Utility/Dataclasses.py`. This is Cython's fallback implementation of dataclasses used when the stdlib module isn't available.

Python's standard library dataclasses documentation: https://docs.python.org/3/library/dataclasses.html

The standard library implementation correctly uses `kw_only` consistently in both the attribute name and repr output. Cython's documentation states it supports "extension types that behave like the dataclasses defined in the Python 3.7+ standard library", implying behavior should match.

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