# Bug Report: scipy.constants.find Cannot Find 90 Constants

**Target**: `scipy.constants.find`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `find()` function searches only the current CODATA 2022 constants but users can access constants from all CODATA versions (2002-2022) through `physical_constants`. This means 90 constants in `physical_constants` cannot be discovered using `find()`, violating the documented purpose of the function.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import scipy.constants as sc


def test_find_none_returns_all_keys():
    result = sc.find(None)
    assert result == sorted(sc.physical_constants.keys())


@given(st.sampled_from(list(sc.physical_constants.keys())))
def test_find_can_locate_all_constants(key):
    results = sc.find(key)
    assert key in results, f"find() could not locate key '{key}' that exists in physical_constants"
```

**Failing input**: `'Planck constant over 2 pi'` (and 89 other constants from older CODATA versions)

## Reproducing the Bug

```python
import scipy.constants as sc

key = 'Planck constant over 2 pi'

print(f"sc.physical_constants['{key}'] = {sc.physical_constants[key]}")

results = sc.find('Planck constant over 2 pi')
print(f"sc.find('Planck constant over 2 pi') = {results}")

assert len(results) == 0, "find() cannot locate this constant!"

print(f"\n_current_constants has {len(sc._codata._current_constants)} keys")
print(f"physical_constants has {len(sc.physical_constants)} keys")
print(f"{len(sc.physical_constants) - len(sc._codata._current_constants)} constants are inaccessible via find()")
```

## Why This Is A Bug

The function documentation states: "Return list of physical_constant keys containing a given string." Users reasonably expect `find()` to search all keys in `physical_constants`, not just a subset. This violates the API contract because:

1. The docstring doesn't mention it only searches current CODATA constants
2. `physical_constants` is the public API for accessing constants
3. 90 constants (20% of the total) are accessible via `physical_constants` but undiscoverable via `find()`
4. The example in the docstring shows `find()` being used with `physical_constants`, implying they're consistent

## Fix

```diff
--- a/scipy/constants/_codata.py
+++ b/scipy/constants/_codata.py
@@ -2248,9 +2248,9 @@ def find(sub: str | None = None, disp: bool = False) -> Any:

     """
     if sub is None:
-        result = list(_current_constants.keys())
+        result = list(physical_constants.keys())
     else:
-        result = [key for key in _current_constants
+        result = [key for key in physical_constants
                   if sub.lower() in key.lower()]

     result.sort()
```