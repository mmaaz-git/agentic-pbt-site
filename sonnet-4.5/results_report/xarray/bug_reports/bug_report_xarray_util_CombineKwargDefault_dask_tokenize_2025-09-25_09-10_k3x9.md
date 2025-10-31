# Bug Report: xarray.util CombineKwargDefault Mutable Dask Token

**Target**: `xarray.util.deprecation_helpers.CombineKwargDefault.__dask_tokenize__`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `CombineKwargDefault.__dask_tokenize__` method returns different tokens based on global OPTIONS, violating dask's requirement that an object's token must remain constant for proper caching and memoization.

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
```

**Failing input**: Any `CombineKwargDefault` with `old != new`, e.g., `name='0'`

## Reproducing the Bug

```python
from xarray.util.deprecation_helpers import CombineKwargDefault
from xarray.core.options import set_options

obj = CombineKwargDefault(name="test", old="old_value", new="new_value")

with set_options(use_new_combine_kwarg_defaults=False):
    token1 = obj.__dask_tokenize__()
    print(f"Token with use_new=False: {token1}")

with set_options(use_new_combine_kwarg_defaults=True):
    token2 = obj.__dask_tokenize__()
    print(f"Token with use_new=True: {token2}")

print(f"\nTokens are equal: {token1 == token2}")
```

Output:
```
Token with use_new=False: ('tuple', (('c2bacaeba7fcaff398dc932633c8c187107cfc52', []), 'old_value'))
Token with use_new=True: ('tuple', (('c2bacaeba7fcaff398dc932633c8c187107cfc52', []), 'new_value'))

Tokens are equal: False
```

## Why This Is A Bug

Dask uses `__dask_tokenize__` to generate cache keys and determine when computations can be reused. When the same object produces different tokens at different times, it breaks dask's caching mechanism:

1. Dask caches computation results based on tokens
2. When OPTIONS change, the same object gets a different token
3. Dask treats it as a different input, invalidating cached results
4. This causes unnecessary recomputation and potential inconsistencies

The current implementation computes the token from `self._value`, which changes based on `OPTIONS["use_new_combine_kwarg_defaults"]`. This is inconsistent with the object's identity.

This bug is related to the hash immutability bug found in the same class, but affects dask operations specifically.

## Fix

The dask token should be based on immutable attributes only, just like the hash fix:

```diff
--- a/xarray/util/deprecation_helpers.py
+++ b/xarray/util/deprecation_helpers.py
@@ -183,7 +183,7 @@ class CombineKwargDefault:
     def __dask_tokenize__(self) -> object:
         from dask.base import normalize_token

-        return normalize_token((type(self), self._value))
+        return normalize_token((type(self), self._name, self._old, self._new))
```

This ensures the dask token remains constant for the object's lifetime, consistent with its immutable attributes rather than its mutable value.