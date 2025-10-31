# Bug Report: CombineKwargDefault Hash Instability Violates Python's Hash Contract

**Target**: `xarray.util.deprecation_helpers.CombineKwargDefault`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `CombineKwargDefault` class violates Python's fundamental requirement that an object's hash must remain constant during its lifetime - the hash changes when the global `OPTIONS["use_new_combine_kwarg_defaults"]` setting is modified.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from xarray.util.deprecation_helpers import CombineKwargDefault
from xarray.core.options import OPTIONS


@given(st.sampled_from(["all", "minimal", "exact"]))
@settings(max_examples=100)
def test_hash_stability_across_options_change(val):
    obj = CombineKwargDefault(name="test", old="old_value", new="new_value")

    original_option = OPTIONS["use_new_combine_kwarg_defaults"]

    try:
        OPTIONS["use_new_combine_kwarg_defaults"] = False
        hash1 = hash(obj)

        OPTIONS["use_new_combine_kwarg_defaults"] = True
        hash2 = hash(obj)

        assert hash1 == hash2
    finally:
        OPTIONS["use_new_combine_kwarg_defaults"] = original_option
```

<details>

<summary>
**Failing input**: `val='all'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 26, in <module>
    test_hash_stability_across_options_change()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 7, in test_hash_stability_across_options_change
    @settings(max_examples=100)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 20, in test_hash_stability_across_options_change
    assert hash1 == hash2
           ^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_hash_stability_across_options_change(
    val='all',
)
```
</details>

## Reproducing the Bug

```python
from xarray.util.deprecation_helpers import CombineKwargDefault
from xarray.core.options import OPTIONS

# Create a CombineKwargDefault object with different old/new values
obj = CombineKwargDefault(name="test", old="old_value", new="new_value")

# Set global option to False and add object to a set
OPTIONS["use_new_combine_kwarg_defaults"] = False
print(f"Initial OPTIONS['use_new_combine_kwarg_defaults'] = {OPTIONS['use_new_combine_kwarg_defaults']}")
print(f"obj._value = {obj._value}")
hash_before = hash(obj)
print(f"Hash before: {hash_before}")

# Add the object to a set
s = {obj}
print(f"Object added to set: {obj in s}")

# Change the global option
OPTIONS["use_new_combine_kwarg_defaults"] = True
print(f"\nChanged OPTIONS['use_new_combine_kwarg_defaults'] = {OPTIONS['use_new_combine_kwarg_defaults']}")
print(f"obj._value = {obj._value}")
hash_after = hash(obj)
print(f"Hash after: {hash_after}")

# Check if the object is still in the set
print(f"Object still in set: {obj in s}")

# Verify the hashes are different
print(f"\nHashes are equal: {hash_before == hash_after}")

# Demonstrate the issue - object can't be found in set after hash change
print(f"\nAssertion check: obj in s")
assert obj in s, "Object not found in set after hash changed!"
```

<details>

<summary>
Object disappears from set after global option changes
</summary>
```
Initial OPTIONS['use_new_combine_kwarg_defaults'] = False
obj._value = old_value
Hash before: 65400547969936630
Object added to set: True

Changed OPTIONS['use_new_combine_kwarg_defaults'] = True
obj._value = new_value
Hash after: 5947018016416215716
Object still in set: False

Hashes are equal: False

Assertion check: obj in s
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/39/repo.py", line 33, in <module>
    assert obj in s, "Object not found in set after hash changed!"
           ^^^^^^^^
AssertionError: Object not found in set after hash changed!
```
</details>

## Why This Is A Bug

This violates Python's fundamental hash contract documented in the [Python data model](https://docs.python.org/3/reference/datamodel.html#object.__hash__): "An object is hashable if it has a hash value which never changes during its lifetime."

The bug occurs because:
1. The `CombineKwargDefault.__hash__()` method returns `hash(self._value)` (line 181 in deprecation_helpers.py)
2. The `_value` property dynamically returns either `self._old` or `self._new` based on the global `OPTIONS["use_new_combine_kwarg_defaults"]` setting (line 178)
3. When this global option changes, the hash changes, causing objects to become unfindable in sets/dicts

This breaks core Python semantics - objects added to a set literally disappear when the global option changes, even though `obj in s` should always return `True` for an object that was previously added.

## Relevant Context

The `CombineKwargDefault` class is a helper for managing deprecation cycles in xarray, allowing gradual transitions from old to new default parameter values. While it's primarily an internal utility class, it explicitly implements `__hash__()` and `__eq__()`, signaling its intent to be hashable.

The class is used to define default values for combine operations:
- `_DATA_VARS_DEFAULT` (line 215)
- `_COORDS_DEFAULT` (line 216)
- `_COMPAT_CONCAT_DEFAULT` (line 217-219)
- `_COMPAT_DEFAULT` (line 220)
- `_JOIN_DEFAULT` (line 221)

The global option `use_new_combine_kwarg_defaults` defaults to `False` (line 92 in options.py) and controls whether xarray uses old or new default values during the deprecation transition period.

## Proposed Fix

Base the hash on immutable object properties rather than the mutable `_value`:

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