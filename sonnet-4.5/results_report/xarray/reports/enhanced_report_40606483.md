# Bug Report: xarray.util.deprecation_helpers.CombineKwargDefault Hash Mutability Violation

**Target**: `xarray.util.deprecation_helpers.CombineKwargDefault`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The hash value of `CombineKwargDefault` objects changes when the global `OPTIONS["use_new_combine_kwarg_defaults"]` setting is modified, violating Python's requirement that object hashes must remain constant during their lifetime.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from xarray.util.deprecation_helpers import CombineKwargDefault
from xarray.core.options import set_options

@given(
    name=st.text(min_size=1),
    old=st.text(),
    new=st.text()
)
def test_hash_should_not_change_with_options(name, old, new):
    assume(old != new)

    obj = CombineKwargDefault(name=name, old=old, new=new)

    with set_options(use_new_combine_kwarg_defaults=False):
        hash1 = hash(obj)

    with set_options(use_new_combine_kwarg_defaults=True):
        hash2 = hash(obj)

    assert hash1 == hash2, f"Hash changed from {hash1} to {hash2} for name='{name}', old='{old}', new='{new}'"

if __name__ == "__main__":
    # Run the property test
    test_hash_should_not_change_with_options()
```

<details>

<summary>
**Failing input**: `name='0', old='', new='0'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 25, in <module>
    test_hash_should_not_change_with_options()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 6, in test_hash_should_not_change_with_options
    name=st.text(min_size=1),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/43/hypo.py", line 21, in test_hash_should_not_change_with_options
    assert hash1 == hash2, f"Hash changed from {hash1} to {hash2} for name='{name}', old='{old}', new='{new}'"
           ^^^^^^^^^^^^^^
AssertionError: Hash changed from 0 to -9069594978772472686 for name='0', old='', new='0'
Falsifying example: test_hash_should_not_change_with_options(
    # The test always failed when commented parts were varied together.
    name='0',  # or any other generated value
    old='',  # or any other generated value
    new='0',  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from xarray.util.deprecation_helpers import CombineKwargDefault
from xarray.core.options import set_options

# Create a CombineKwargDefault object
obj = CombineKwargDefault(name="test", old="old_value", new="new_value")

# Test 1: Hash changes with global OPTIONS
print("Test 1: Hash mutation with global OPTIONS")
print("-" * 50)

with set_options(use_new_combine_kwarg_defaults=False):
    hash1 = hash(obj)
    print(f"Hash with use_new_combine_kwarg_defaults=False: {hash1}")

with set_options(use_new_combine_kwarg_defaults=True):
    hash2 = hash(obj)
    print(f"Hash with use_new_combine_kwarg_defaults=True: {hash2}")

print(f"Hash changed: {hash1 != hash2}")
print()

# Test 2: Set membership breaks
print("Test 2: Set membership failure")
print("-" * 50)

with set_options(use_new_combine_kwarg_defaults=False):
    s = {obj}
    print(f"Object added to set: {obj in s}")

with set_options(use_new_combine_kwarg_defaults=True):
    print(f"Object in set after option change: {obj in s}")
print()

# Test 3: Dictionary key lookup fails
print("Test 3: Dictionary key lookup failure")
print("-" * 50)

with set_options(use_new_combine_kwarg_defaults=False):
    d = {obj: "value"}
    print(f"Object added as dict key: {obj in d}")

with set_options(use_new_combine_kwarg_defaults=True):
    try:
        value = d[obj]
        print(f"Successfully retrieved value: {value}")
    except KeyError:
        print("KeyError: Object cannot be found as dictionary key after option change")
print()

# Test 4: Show the failing case from the report
print("Test 4: Specific failing case from report")
print("-" * 50)

obj2 = CombineKwargDefault(name='0', old='', new='0')

with set_options(use_new_combine_kwarg_defaults=False):
    hash3 = hash(obj2)
    print(f"Hash with False (old=''): {hash3}")

with set_options(use_new_combine_kwarg_defaults=True):
    hash4 = hash(obj2)
    print(f"Hash with True (new='0'): {hash4}")

print(f"Hash changed: {hash3 != hash4}")
```

<details>

<summary>
Hash changes and causes set/dict lookup failures
</summary>
```
Test 1: Hash mutation with global OPTIONS
--------------------------------------------------
Hash with use_new_combine_kwarg_defaults=False: 6630480770934499062
Hash with use_new_combine_kwarg_defaults=True: 4046865338554893626
Hash changed: True

Test 2: Set membership failure
--------------------------------------------------
Object added to set: True
Object in set after option change: False

Test 3: Dictionary key lookup failure
--------------------------------------------------
Object added as dict key: True
KeyError: Object cannot be found as dictionary key after option change

Test 4: Specific failing case from report
--------------------------------------------------
Hash with False (old=''): 0
Hash with True (new='0'): 2503097188848615641
Hash changed: True
```
</details>

## Why This Is A Bug

This violates Python's fundamental requirement that hash values must remain constant during an object's lifetime. According to the Python documentation (https://docs.python.org/3/reference/datamodel.html#object.__hash__), objects used in sets or as dictionary keys must have immutable hash values.

The current implementation in `deprecation_helpers.py` (lines 176-181) makes the hash dependent on a mutable property:

```python
@property
def _value(self) -> str | None:
    return self._new if OPTIONS["use_new_combine_kwarg_defaults"] else self._old

def __hash__(self) -> int:
    return hash(self._value)
```

The `_value` property returns different strings based on the global `OPTIONS["use_new_combine_kwarg_defaults"]` setting. When this option changes, the hash changes, causing:

1. **Set membership failures**: Objects added to a set cannot be found after the option changes
2. **Dictionary lookup failures**: Objects used as dictionary keys raise KeyError after the option changes
3. **Silent data corruption**: Collections silently lose track of these objects

While `CombineKwargDefault` objects are primarily used internally for deprecation handling of default parameter values and may not commonly be stored in sets or dictionaries by end users, any code that does store them will experience these failures.

## Relevant Context

The `CombineKwargDefault` class is used throughout xarray to handle the deprecation cycle of default parameter values in functions like `concat`, `merge`, and `combine`. Instances are created as module-level constants (lines 215-222):

```python
_DATA_VARS_DEFAULT = CombineKwargDefault(name="data_vars", old="all", new=None)
_COORDS_DEFAULT = CombineKwargDefault(name="coords", old="different", new="minimal")
_COMPAT_CONCAT_DEFAULT = CombineKwargDefault(name="compat", old="equals", new="override")
_COMPAT_DEFAULT = CombineKwargDefault(name="compat", old="no_conflicts", new="override")
_JOIN_DEFAULT = CombineKwargDefault(name="join", old="outer", new="exact")
```

These objects implement `__eq__` (lines 169-174) and `__hash__` (lines 180-181), indicating they are designed to be hashable. The class also implements `__dask_tokenize__` for dask integration, which similarly uses the mutable `_value` property.

The root cause is that the hash is computed from `_value`, which changes based on global state, rather than from the immutable instance attributes (`_name`, `_old`, `_new`) that define the object's identity.

## Proposed Fix

The hash should be based on the immutable instance attributes that define the object's identity, not on the mutable `_value` property:

```diff
diff --git a/xarray/util/deprecation_helpers.py b/xarray/util/deprecation_helpers.py
index abc1234..def5678 100644
--- a/xarray/util/deprecation_helpers.py
+++ b/xarray/util/deprecation_helpers.py
@@ -177,8 +177,8 @@ class CombineKwargDefault:
     def _value(self) -> str | None:
         return self._new if OPTIONS["use_new_combine_kwarg_defaults"] else self._old

-    def __hash__(self) -> int:
-        return hash(self._value)
+    def __hash__(self) -> int:
+        return hash((self._name, self._old, self._new))

     def __dask_tokenize__(self) -> object:
         from dask.base import normalize_token
```