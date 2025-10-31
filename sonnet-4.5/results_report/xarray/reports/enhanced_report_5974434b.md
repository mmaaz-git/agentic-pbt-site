# Bug Report: xarray.util.deprecation_helpers.CombineKwargDefault Hash Invariant Violation

**Target**: `xarray.util.deprecation_helpers.CombineKwargDefault`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `CombineKwargDefault.__hash__()` method violates Python's hash invariant by returning different hash values for the same object when the global `OPTIONS["use_new_combine_kwarg_defaults"]` setting changes, causing dictionary key lookups to fail.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from xarray.util.deprecation_helpers import CombineKwargDefault
from xarray.core.options import OPTIONS


@given(
    name=st.text(min_size=1),
    old=st.text(),
    new=st.text(),
)
@settings(max_examples=100)
def test_hash_remains_constant_during_object_lifetime(name, old, new):
    """
    Property: An object's hash must remain constant during its lifetime.
    """
    obj = CombineKwargDefault(name=name, old=old, new=new)

    original_hash = hash(obj)
    original_option = OPTIONS["use_new_combine_kwarg_defaults"]

    OPTIONS["use_new_combine_kwarg_defaults"] = not original_option
    new_hash = hash(obj)
    OPTIONS["use_new_combine_kwarg_defaults"] = original_option

    assert original_hash == new_hash, (
        f"Hash changed when global OPTIONS changed! "
        f"Before: {original_hash}, After: {new_hash}. "
        f"This violates Python's hash invariant."
    )


if __name__ == "__main__":
    test_hash_remains_constant_during_object_lifetime()
```

<details>

<summary>
**Failing input**: `name='0', old='', new='0'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 33, in <module>
    test_hash_remains_constant_during_object_lifetime()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 7, in test_hash_remains_constant_during_object_lifetime
    name=st.text(min_size=1),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/50/hypo.py", line 25, in test_hash_remains_constant_during_object_lifetime
    assert original_hash == new_hash, (
           ^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Hash changed when global OPTIONS changed! Before: 0, After: -1391512760308981013. This violates Python's hash invariant.
Falsifying example: test_hash_remains_constant_during_object_lifetime(
    name='0',  # or any other generated value
    old='',  # or any other generated value
    new='0',
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/50/hypo.py:26
```
</details>

## Reproducing the Bug

```python
from xarray.util.deprecation_helpers import CombineKwargDefault
from xarray.core.options import OPTIONS

# Create a CombineKwargDefault object with different old and new values
obj = CombineKwargDefault(name="test", old="old_val", new="new_val")

# Store the object as a dictionary key
d = {obj: "stored_value"}

# Initial retrieval - this should work
print(f"Initial OPTIONS['use_new_combine_kwarg_defaults']: {OPTIONS['use_new_combine_kwarg_defaults']}")
print(f"Initial hash of object: {hash(obj)}")
print(f"Initial value from dict: {d[obj]}")
print()

# Change the global OPTIONS setting
OPTIONS["use_new_combine_kwarg_defaults"] = True
print(f"Changed OPTIONS['use_new_combine_kwarg_defaults'] to: {OPTIONS['use_new_combine_kwarg_defaults']}")
print(f"New hash of object: {hash(obj)}")

# Try to retrieve the value again - this will fail
try:
    print(f"Value from dict after OPTIONS change: {d[obj]}")
except KeyError as e:
    print(f"KeyError: Object lost! Hash changed, so dict lookup fails.")
    print(f"The key {obj} is no longer found in the dictionary.")
```

<details>

<summary>
KeyError when accessing dictionary after OPTIONS change
</summary>
```
Initial OPTIONS['use_new_combine_kwarg_defaults']: False
Initial hash of object: 2712181757414178173
Initial value from dict: stored_value

Changed OPTIONS['use_new_combine_kwarg_defaults'] to: True
New hash of object: -3003431582126912833
KeyError: Object lost! Hash changed, so dict lookup fails.
The key new_val is no longer found in the dictionary.
```
</details>

## Why This Is A Bug

This violates Python's fundamental hash invariant documented in the [Python data model](https://docs.python.org/3/reference/datamodel.html#object.__hash__): "the hash value should remain constant during an object's lifetime."

The bug occurs because:

1. `CombineKwargDefault.__hash__()` returns `hash(self._value)` (line 181 in deprecation_helpers.py)
2. `_value` is a property that dynamically returns either `self._old` or `self._new` based on the global `OPTIONS["use_new_combine_kwarg_defaults"]` setting (line 177-178)
3. When this global option toggles, existing objects' hash values change

This causes severe issues in practice:
- Objects stored as dictionary keys become inaccessible after OPTIONS changes
- Objects in sets can no longer be found or removed
- Any hash-based data structure will exhibit unpredictable behavior
- Silent data loss or corruption is possible in code relying on these objects as keys

While the CombineKwargDefault class is primarily used internally for deprecation management and not typically used as dictionary keys in the xarray codebase, the implementation of `__hash__()` makes it a hashable type that must follow Python's rules.

## Relevant Context

The `CombineKwargDefault` class is used to manage deprecation transitions for keyword argument defaults in xarray. It allows the library to switch default values based on a global configuration option. The class is instantiated in several places (lines 215-221 of deprecation_helpers.py) to create default values like:
- `_DATA_VARS_DEFAULT`
- `_COORDS_DEFAULT`
- `_COMPAT_CONCAT_DEFAULT`
- `_COMPAT_DEFAULT`
- `_JOIN_DEFAULT`

These instances are used as default parameter values in functions like `concat()`, `merge()`, and `combine_by_coords()`. The global option `OPTIONS["use_new_combine_kwarg_defaults"]` (default: False) controls whether old or new defaults are used.

The class inherits conceptual design from `ReprObject` and implements `__eq__`, `__repr__`, `__hash__`, and `__dask_tokenize__` methods to make it behave like a value type.

## Proposed Fix

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

This fix ensures the hash is based on the immutable attributes that define the object's identity (`_name`, `_old`, `_new`) rather than the dynamic `_value` property that can change based on global state.