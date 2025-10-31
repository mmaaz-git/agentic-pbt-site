# Bug Report: xarray.util.deprecation_helpers.CombineKwargDefault Hash Mutability Violation

**Target**: `xarray.util.deprecation_helpers.CombineKwargDefault`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `CombineKwargDefault.__hash__` method violates Python's hash invariant by returning different hash values for the same object when global OPTIONS state changes, causing objects to become unretrievable from sets and dictionaries.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from xarray.util.deprecation_helpers import CombineKwargDefault
from xarray.core.options import set_options


@given(
    name=st.text(min_size=1),
    old=st.text(),
    new=st.one_of(st.none(), st.text())
)
@settings(max_examples=1000)
def test_combine_kwarg_default_hash_immutable(name, old, new):
    obj = CombineKwargDefault(name=name, old=old, new=new)

    hash1 = hash(obj)

    with set_options(use_new_combine_kwarg_defaults=True):
        hash2 = hash(obj)

    with set_options(use_new_combine_kwarg_defaults=False):
        hash3 = hash(obj)

    assert hash1 == hash2 == hash3, f"Hash changed! hash1={hash1}, hash2={hash2}, hash3={hash3}"


if __name__ == "__main__":
    test_combine_kwarg_default_hash_immutable()
```

<details>

<summary>
**Failing input**: `name='0', old='', new=None`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/20/hypo.py", line 27, in <module>
    test_combine_kwarg_default_hash_immutable()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/20/hypo.py", line 7, in test_combine_kwarg_default_hash_immutable
    name=st.text(min_size=1),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/20/hypo.py", line 23, in test_combine_kwarg_default_hash_immutable
    assert hash1 == hash2 == hash3, f"Hash changed! hash1={hash1}, hash2={hash2}, hash3={hash3}"
           ^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Hash changed! hash1=0, hash2=4238894112, hash3=0
Falsifying example: test_combine_kwarg_default_hash_immutable(
    # The test sometimes passed when commented parts were varied together.
    name='0',  # or any other generated value
    old='',  # or any other generated value
    new=None,
)
```
</details>

## Reproducing the Bug

```python
from xarray.util.deprecation_helpers import CombineKwargDefault
from xarray.core.options import set_options

# Create a CombineKwargDefault object with different old and new values
obj = CombineKwargDefault(name="test", old="old_val", new="new_val")

# Get initial hash
hash1 = hash(obj)
print(f"Initial hash: {hash1}")

# Change OPTIONS and get hash again
with set_options(use_new_combine_kwarg_defaults=True):
    hash2 = hash(obj)
    print(f"Hash after set_options(use_new_combine_kwarg_defaults=True): {hash2}")

# Back to original OPTIONS
with set_options(use_new_combine_kwarg_defaults=False):
    hash3 = hash(obj)
    print(f"Hash after set_options(use_new_combine_kwarg_defaults=False): {hash3}")

print(f"\nAll hashes equal: {hash1 == hash2 == hash3}")

# Demonstrate the practical issue: object lost in set
print("\n--- Demonstrating set/dict issue ---")
s = {obj}
print(f"Object added to set: {obj in s}")

with set_options(use_new_combine_kwarg_defaults=True):
    print(f"Object in set after OPTIONS change: {obj in s}")

# Demonstrate the issue with dictionaries too
print("\n--- Demonstrating dictionary issue ---")
d = {obj: "value"}
print(f"Object used as dict key: {obj in d}")

with set_options(use_new_combine_kwarg_defaults=True):
    print(f"Object as dict key after OPTIONS change: {obj in d}")
```

<details>

<summary>
Hash values change with OPTIONS, causing objects to be lost in containers
</summary>
```
Initial hash: 8946792849879116524
Hash after set_options(use_new_combine_kwarg_defaults=True): -8364521095578338832
Hash after set_options(use_new_combine_kwarg_defaults=False): 8946792849879116524

All hashes equal: False

--- Demonstrating set/dict issue ---
Object added to set: True
Object in set after OPTIONS change: False

--- Demonstrating dictionary issue ---
Object used as dict key: True
Object as dict key after OPTIONS change: False
```
</details>

## Why This Is A Bug

This implementation violates Python's fundamental hash contract as documented in the Python data model (https://docs.python.org/3/reference/datamodel.html#object.__hash__):

> "The only required property is that objects which compare equal have the same hash value"

And more importantly, the implicit but universally understood requirement that an object's hash value must remain constant during its lifetime. This is why mutable objects like lists don't implement `__hash__`.

The violation occurs because both `__hash__` and `__eq__` depend on the `_value` property, which dynamically reads from the global `OPTIONS` dictionary:

```python
@property
def _value(self) -> str | None:
    return self._new if OPTIONS["use_new_combine_kwarg_defaults"] else self._old

def __hash__(self) -> int:
    return hash(self._value)
```

When `OPTIONS["use_new_combine_kwarg_defaults"]` changes from `False` to `True`, the object's `_value` switches from `self._old` to `self._new`, causing the hash to change. This breaks hash-based containers: objects become unretrievable from sets and dictionary keys fail to match, as demonstrated in the reproduction.

## Relevant Context

The `CombineKwargDefault` class is used internally by xarray to handle deprecation cycles for keyword argument default values. It's instantiated as module-level constants in `/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages/xarray/util/deprecation_helpers.py:215-221`:

```python
_DATA_VARS_DEFAULT = CombineKwargDefault(name="data_vars", old="all", new=None)
_COORDS_DEFAULT = CombineKwargDefault(name="coords", old="different", new="minimal")
_COMPAT_CONCAT_DEFAULT = CombineKwargDefault(name="compat", old="equals", new="override")
_COMPAT_DEFAULT = CombineKwargDefault(name="compat", old="no_conflicts", new="override")
_JOIN_DEFAULT = CombineKwargDefault(name="join", old="outer", new="exact")
```

These objects are used as default parameter values in various xarray functions to smoothly transition default behaviors between versions. While not typically used directly by end users in sets or dicts, the hash implementation should still be correct according to Python's data model.

## Proposed Fix

The hash should be based on the immutable identity attributes (`_name`, `_old`, `_new`) rather than the mutable `_value` property:

```diff
--- a/xarray/util/deprecation_helpers.py
+++ b/xarray/util/deprecation_helpers.py
@@ -178,7 +178,7 @@ class CombineKwargDefault:
         return self._new if OPTIONS["use_new_combine_kwarg_defaults"] else self._old

     def __hash__(self) -> int:
-        return hash(self._value)
+        return hash((self._name, self._old, self._new))

     def __dask_tokenize__(self) -> object:
         from dask.base import normalize_token
```