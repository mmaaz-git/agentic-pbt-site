# Bug Report: xarray.util.deprecation_helpers.CombineKwargDefault Mutable Hash and Dask Token

**Target**: `xarray.util.deprecation_helpers.CombineKwargDefault.__dask_tokenize__` and `xarray.util.deprecation_helpers.CombineKwargDefault.__hash__`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `CombineKwargDefault` class violates both Python's hash immutability requirement and Dask's token determinism requirement by basing these values on mutable state that changes with global OPTIONS settings.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
from xarray.util.deprecation_helpers import CombineKwargDefault
from xarray.core.options import set_options


@given(st.text(min_size=1, max_size=50))
@settings(max_examples=200)
def test_dask_tokenize_immutability(name):
    obj = CombineKwargDefault(name=name, old="old_value", new="new_value")

    with set_options(use_new_combine_kwarg_defaults=False):
        token1 = obj.__dask_tokenize__()

    with set_options(use_new_combine_kwarg_defaults=True):
        token2 = obj.__dask_tokenize__()

    assert token1 == token2, f"Dask token changed when OPTIONS changed: {token1} != {token2}"


if __name__ == "__main__":
    test_dask_tokenize_immutability()
```

<details>

<summary>
**Failing input**: `name='0'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 21, in <module>
    test_dask_tokenize_immutability()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 7, in test_dask_tokenize_immutability
    @settings(max_examples=200)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 17, in test_dask_tokenize_immutability
    assert token1 == token2, f"Dask token changed when OPTIONS changed: {token1} != {token2}"
           ^^^^^^^^^^^^^^^^
AssertionError: Dask token changed when OPTIONS changed: ('tuple', (('c2bacaeba7fcaff398dc932633c8c187107cfc52', []), 'old_value')) != ('tuple', (('c2bacaeba7fcaff398dc932633c8c187107cfc52', []), 'new_value'))
Falsifying example: test_dask_tokenize_immutability(
    name='0',
)
```
</details>

## Reproducing the Bug

```python
from xarray.util.deprecation_helpers import CombineKwargDefault
from xarray.core.options import set_options

# Create a CombineKwargDefault object
obj = CombineKwargDefault(name="test", old="old_value", new="new_value")

print("Testing dask tokenization with different OPTIONS settings:")
print("=" * 60)

# Get token with use_new_combine_kwarg_defaults=False
with set_options(use_new_combine_kwarg_defaults=False):
    token1 = obj.__dask_tokenize__()
    hash1 = hash(obj)
    print(f"With use_new=False:")
    print(f"  Token: {token1}")
    print(f"  Hash: {hash1}")

# Get token with use_new_combine_kwarg_defaults=True
with set_options(use_new_combine_kwarg_defaults=True):
    token2 = obj.__dask_tokenize__()
    hash2 = hash(obj)
    print(f"\nWith use_new=True:")
    print(f"  Token: {token2}")
    print(f"  Hash: {hash2}")

print("\n" + "=" * 60)
print(f"Tokens are equal: {token1 == token2}")
print(f"Hashes are equal: {hash1 == hash2}")

if token1 != token2:
    print("\nERROR: Dask tokens changed when OPTIONS changed!")
    print("This violates dask's requirement that tokens must be deterministic.")

if hash1 != hash2:
    print("\nERROR: Hash values changed when OPTIONS changed!")
    print("This violates Python's requirement that hash must be constant during an object's lifetime.")
```

<details>

<summary>
Output demonstrating both hash and dask token violations
</summary>
```
Testing dask tokenization with different OPTIONS settings:
============================================================
With use_new=False:
  Token: ('tuple', (('c2bacaeba7fcaff398dc932633c8c187107cfc52', []), 'old_value'))
  Hash: 4880690391740102313

With use_new=True:
  Token: ('tuple', (('c2bacaeba7fcaff398dc932633c8c187107cfc52', []), 'new_value'))
  Hash: -4800675194723377181

============================================================
Tokens are equal: False
Hashes are equal: False

ERROR: Dask tokens changed when OPTIONS changed!
This violates dask's requirement that tokens must be deterministic.

ERROR: Hash values changed when OPTIONS changed!
This violates Python's requirement that hash must be constant during an object's lifetime.
```
</details>

## Why This Is A Bug

This implementation violates two fundamental contracts:

1. **Python's hash immutability requirement**: According to Python documentation, "the hash value of an object must never change during its lifetime." The current implementation computes hash from `self._value` (line 181), which changes based on `OPTIONS["use_new_combine_kwarg_defaults"]`. This can lead to:
   - Dictionary/set corruption when the object is used as a key
   - Objects "disappearing" from collections after OPTIONS change
   - Undefined behavior in any hash-based data structure

2. **Dask's token determinism requirement**: Dask documentation explicitly states that `__dask_tokenize__` must be "idempotent" and "deterministic" - the same object must always produce the same token. The current implementation (line 186) violates this by returning different tokens when global OPTIONS change. This causes:
   - Cache invalidation when OPTIONS change
   - Unnecessary recomputation of cached results
   - Potential inconsistencies in distributed computations
   - Performance degradation due to broken memoization

The root cause is that both methods use `self._value` property (line 177-178), which returns different values (`self._old` or `self._new`) depending on the global `OPTIONS["use_new_combine_kwarg_defaults"]` setting. This makes the object's identity mutable from the perspective of hashing and tokenization.

## Relevant Context

The `CombineKwargDefault` class is part of xarray's internal machinery for handling deprecation cycles when changing default keyword argument values. It's instantiated at module level for several defaults (lines 215-221 in deprecation_helpers.py):

- `_DATA_VARS_DEFAULT`: transitions from `"all"` to `None`
- `_COORDS_DEFAULT`: transitions from `"different"` to `"minimal"`
- `_COMPAT_CONCAT_DEFAULT`: transitions from `"equals"` to `"override"`
- `_COMPAT_DEFAULT`: transitions from `"no_conflicts"` to `"override"`
- `_JOIN_DEFAULT`: transitions from `"outer"` to `"exact"`

These objects are used throughout xarray to handle the deprecation transition period. While users don't directly interact with these objects, they're used internally in functions that might be called within Dask operations, where the tokenization bug could cause cache invalidation.

The bug is particularly insidious because:
- It only manifests when OPTIONS change during program execution
- The symptoms (cache misses, performance degradation) might not be immediately traced to this cause
- The hash violation could cause hard-to-debug issues in any code that uses these objects as dictionary keys

Documentation references:
- [Python hash requirements](https://docs.python.org/3/reference/datamodel.html#object.__hash__)
- [Dask tokenization documentation](https://docs.dask.org/en/latest/custom-collections.html)

## Proposed Fix

```diff
--- a/xarray/util/deprecation_helpers.py
+++ b/xarray/util/deprecation_helpers.py
@@ -177,10 +177,10 @@ class CombineKwargDefault:
         return self._new if OPTIONS["use_new_combine_kwarg_defaults"] else self._old

     def __hash__(self) -> int:
-        return hash(self._value)
+        return hash((type(self), self._name, self._old, self._new))

     def __dask_tokenize__(self) -> object:
         from dask.base import normalize_token

-        return normalize_token((type(self), self._value))
+        return normalize_token((type(self), self._name, self._old, self._new))
```