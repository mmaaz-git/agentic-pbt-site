# Bug Report: xarray.util.deprecation_helpers.CombineKwargDefault Violates Python Hash Contract

**Target**: `xarray.util.deprecation_helpers.CombineKwargDefault`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `CombineKwargDefault` class violates Python's hash contract by allowing an object's hash value to change during its lifetime when the global `OPTIONS["use_new_combine_kwarg_defaults"]` setting is modified, causing objects to become "lost" in dictionaries and sets.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from xarray.util.deprecation_helpers import CombineKwargDefault
from xarray.core.options import OPTIONS


@given(
    name=st.text(min_size=1),
    old=st.text(),
    new=st.one_of(st.none(), st.text())
)
@settings(max_examples=1000)
def test_combine_kwarg_hash_stable_across_options_change(name, old, new):
    obj = CombineKwargDefault(name=name, old=old, new=new)

    original_setting = OPTIONS["use_new_combine_kwarg_defaults"]
    hash1 = hash(obj)

    OPTIONS["use_new_combine_kwarg_defaults"] = not original_setting
    hash2 = hash(obj)

    OPTIONS["use_new_combine_kwarg_defaults"] = original_setting

    assert hash1 == hash2, f"Hash changed from {hash1} to {hash2} when OPTIONS changed"


# Run the test
if __name__ == "__main__":
    test_combine_kwarg_hash_stable_across_options_change()
```

<details>

<summary>
**Failing input**: `name='0', old='', new=None`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 31, in <module>
    test_combine_kwarg_hash_stable_across_options_change()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 10, in test_combine_kwarg_hash_stable_across_options_change
    name=st.text(min_size=1),
               ^^^
  File "/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 26, in test_combine_kwarg_hash_stable_across_options_change
    assert hash1 == hash2, f"Hash changed from {hash1} to {hash2} when OPTIONS changed"
           ^^^^^^^^^^^^^^
AssertionError: Hash changed from 0 to 4238894112 when OPTIONS changed
Falsifying example: test_combine_kwarg_hash_stable_across_options_change(
    # The test sometimes passed when commented parts were varied together.
    name='0',  # or any other generated value
    old='',  # or any other generated value
    new=None,
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from xarray.util.deprecation_helpers import CombineKwargDefault
from xarray.core.options import OPTIONS

# Create an instance of CombineKwargDefault
obj = CombineKwargDefault(name='test', old='old_val', new='new_val')

# Save original setting
original_setting = OPTIONS["use_new_combine_kwarg_defaults"]

# Get hash with original setting
hash1 = hash(obj)
print(f"Hash with OPTIONS['use_new_combine_kwarg_defaults']={original_setting}: {hash1}")

# Change the global OPTIONS setting
OPTIONS["use_new_combine_kwarg_defaults"] = not original_setting

# Get hash with changed setting
hash2 = hash(obj)
print(f"Hash with OPTIONS['use_new_combine_kwarg_defaults']={not original_setting}: {hash2}")

# Check if hash changed
if hash1 != hash2:
    print("\n❌ BUG: Hash changed when OPTIONS setting changed!")
    print(f"   This violates Python's hash contract.")

    # Demonstrate the practical impact: object gets lost in dictionary
    print("\nDemonstrating dictionary lookup failure:")

    # Reset to original setting
    OPTIONS["use_new_combine_kwarg_defaults"] = original_setting

    # Create dictionary with object as key
    test_dict = {obj: "value"}
    print(f"  Created dict with obj as key: {test_dict}")

    # Change OPTIONS again
    OPTIONS["use_new_combine_kwarg_defaults"] = not original_setting

    # Try to retrieve value - this will fail
    try:
        value = test_dict[obj]
        print(f"  Retrieved value: {value}")
    except KeyError:
        print(f"  ❌ KeyError: Object not found in dictionary after OPTIONS change!")
        print(f"     The object is 'lost' because its hash changed")
else:
    print("✓ Hash remained constant (expected behavior)")

# Reset OPTIONS to original value
OPTIONS["use_new_combine_kwarg_defaults"] = original_setting
```

<details>

<summary>
Hash instability causes dictionary lookup failure
</summary>
```
Hash with OPTIONS['use_new_combine_kwarg_defaults']=False: -3991775583419184047
Hash with OPTIONS['use_new_combine_kwarg_defaults']=True: 8543826041596549827

❌ BUG: Hash changed when OPTIONS setting changed!
   This violates Python's hash contract.

Demonstrating dictionary lookup failure:
  Created dict with obj as key: {old_val: 'value'}
  ❌ KeyError: Object not found in dictionary after OPTIONS change!
     The object is 'lost' because its hash changed
```
</details>

## Why This Is A Bug

This violates Python's fundamental hash contract as documented in the Python Data Model documentation:

> "If a class defines mutable objects and implements an `__eq__()` method, it should not implement `__hash__()`, since the implementation of hashable collections requires that a key's hash value is immutable (if the object's hash value changes, it will be in the wrong hash bucket)."

The `CombineKwargDefault` class implements both `__eq__` and `__hash__` based on the `_value` property, which changes when the global `OPTIONS["use_new_combine_kwarg_defaults"]` setting changes:

```python
@property
def _value(self) -> str | None:
    return self._new if OPTIONS["use_new_combine_kwarg_defaults"] else self._old

def __hash__(self) -> int:
    return hash(self._value)  # Hash depends on mutable _value

def __eq__(self, other: Self | Any) -> bool:
    return (
        self._value == other._value  # Equality also depends on mutable _value
        if isinstance(other, type(self))
        else self._value == other
    )
```

This causes real problems:
1. **Objects disappear from dictionaries**: When used as dictionary keys, objects become unretrievable after OPTIONS changes
2. **Set operations fail**: Objects added to sets may not be found after OPTIONS changes
3. **Cache invalidation issues**: Hash-based caches become unreliable
4. **Unexpected behavior in data structures**: Any data structure relying on hash stability breaks

## Relevant Context

The `CombineKwargDefault` class is used throughout xarray for managing deprecation cycles of keyword argument defaults. Several instances are defined at the module level:

- `_DATA_VARS_DEFAULT`: manages the `data_vars` parameter transition
- `_COORDS_DEFAULT`: manages the `coords` parameter transition
- `_COMPAT_CONCAT_DEFAULT`: manages the `compat` parameter in concat operations
- `_COMPAT_DEFAULT`: manages the `compat` parameter generally
- `_JOIN_DEFAULT`: manages the `join` parameter transition

These objects are used as default values in function signatures and need to maintain consistent identity throughout their lifetime. The hash instability can cause issues when these defaults are stored in caches or used in any hash-based lookup.

The `OPTIONS["use_new_combine_kwarg_defaults"]` setting (defined in `xarray.core.options`) is a global flag that controls whether to use old or new default values during the deprecation transition period.

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

This fix ensures the hash is based on the immutable construction parameters (`_name`, `_old`, `_new`) rather than the mutable `_value` property. This maintains hash stability while still allowing `__eq__` to use `_value` for equality comparisons based on the current OPTIONS setting.