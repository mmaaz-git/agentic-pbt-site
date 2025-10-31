# Bug Report: xarray.util.deprecation_helpers.CombineKwargDefault Hash Instability

**Target**: `xarray.util.deprecation_helpers.CombineKwargDefault`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `CombineKwargDefault` class implements `__hash__` based on a property that depends on global OPTIONS state, causing the object's hash value to change when `set_options(use_new_combine_kwarg_defaults=...)` is called. This violates Python's hash contract and can cause dict/set lookups to fail.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import xarray
from xarray.util.deprecation_helpers import CombineKwargDefault

@given(
    name=st.text(min_size=1, max_size=20),
    old_val=st.text(min_size=1, max_size=10),
    new_val=st.text(min_size=1, max_size=10) | st.none()
)
def test_hash_stability(name, old_val, new_val):
    """Hash of an object should remain constant during its lifetime."""
    obj = CombineKwargDefault(name=name, old=old_val, new=new_val)

    xarray.set_options(use_new_combine_kwarg_defaults=False)
    hash1 = hash(obj)

    xarray.set_options(use_new_combine_kwarg_defaults=True)
    hash2 = hash(obj)

    assert hash1 == hash2, f"Hash changed from {hash1} to {hash2} when options changed"
```

**Failing input**: Any `CombineKwargDefault` object with different old/new values

## Reproducing the Bug

```python
import xarray
from xarray.util.deprecation_helpers import CombineKwargDefault

obj = CombineKwargDefault(name="test", old="old_value", new="new_value")

xarray.set_options(use_new_combine_kwarg_defaults=False)
hash1 = hash(obj)
print(f"Hash with old defaults: {hash1}")

xarray.set_options(use_new_combine_kwarg_defaults=True)
hash2 = hash(obj)
print(f"Hash with new defaults: {hash2}")

assert hash1 == hash2, f"Hash changed: {hash1} != {hash2}"
```

**Output:**
```
Hash with old defaults: <some_value>
Hash with new defaults: <different_value>
AssertionError: Hash changed: <hash1> != <hash2>
```

**Practical impact - dict lookup failure:**
```python
import xarray
from xarray.util.deprecation_helpers import CombineKwargDefault

xarray.set_options(use_new_combine_kwarg_defaults=False)
obj = CombineKwargDefault(name="param", old="old", new="new")
test_dict = {obj: "value"}

xarray.set_options(use_new_combine_kwarg_defaults=True)
print(obj in test_dict)
```

**Output:**
```
False
```
The object cannot be found in the dict even though it was just inserted!

## Why This Is A Bug

1. **Violates Python's Hash Contract**: Python's documentation states: "The hash value of an object should never change during its lifetime (it needs to stay constant)." When OPTIONS change, the hash changes, violating this fundamental invariant.

2. **Breaks Hash-Based Data Structures**: If a `CombineKwargDefault` instance is used as a dict key or stored in a set:
   - Dict lookups will fail if OPTIONS change after insertion
   - Set membership tests will return incorrect results
   - This makes the object unsuitable for use in any hash-based collection

3. **Inconsistent with `__eq__`**: While `__eq__` correctly reflects the current state, `__hash__` must be based on immutable attributes. Two objects that compare equal must have the same hash, but more importantly, an object's hash must not change.

4. **Real-World Impact**: Although these objects are used internally by xarray, any code that stores them in dicts/sets (e.g., for caching, deduplication) will experience silent failures when global options change.

## Root Cause

The bug is in the `__hash__` implementation at line 180-181 of `xarray/util/deprecation_helpers.py`:

```python
def __hash__(self) -> int:
    return hash(self._value)  # _value is a property that reads OPTIONS!
```

The `_value` property (line 176-178) returns different values based on global state:

```python
@property
def _value(self) -> str | None:
    return self._new if OPTIONS["use_new_combine_kwarg_defaults"] else self._old
```

## Fix

The hash should be based on the immutable instance attributes, not the computed `_value`:

```diff
diff --git a/xarray/util/deprecation_helpers.py b/xarray/util/deprecation_helpers.py
index 1234567..abcdefg 100644
--- a/xarray/util/deprecation_helpers.py
+++ b/xarray/util/deprecation_helpers.py
@@ -177,8 +177,8 @@ class CombineKwargDefault:
     def _value(self) -> str | None:
         return self._new if OPTIONS["use_new_combine_kwarg_defaults"] else self._old

     def __hash__(self) -> int:
-        return hash(self._value)
+        return hash((self._name, self._old, self._new))
```

This fix ensures:
1. The hash is based only on immutable attributes set at initialization
2. The hash remains constant throughout the object's lifetime
3. Two instances with the same name/old/new will always have the same hash
4. `__eq__` and `__hash__` remain consistent (equal objects have equal hashes)