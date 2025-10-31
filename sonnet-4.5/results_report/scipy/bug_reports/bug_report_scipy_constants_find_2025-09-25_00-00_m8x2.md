# Bug Report: scipy.constants.find() Doesn't Return All Keys

**Target**: `scipy.constants.find()`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `find()` function claims in its docstring to "By default, return all keys" when called with `None`, but it only returns keys from the current CODATA version (355 keys) instead of all keys in `physical_constants` (445 keys), missing 90 constants.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
import scipy.constants as sc

def test_find_none_returns_all_keys():
    results = sc.find(None)
    assert set(results) == set(sc.physical_constants.keys())

def test_find_empty_string_returns_all_keys():
    results = sc.find('')
    assert set(results) == set(sc.physical_constants.keys())
```

**Failing input**: Both `find(None)` and `find('')` fail the assertion.

## Reproducing the Bug

```python
import scipy.constants as sc

all_keys = set(sc.physical_constants.keys())
find_none_keys = set(sc.find(None))
find_empty_keys = set(sc.find(''))

print(f"Total keys in physical_constants: {len(all_keys)}")
print(f"Keys returned by find(None): {len(find_none_keys)}")
print(f"Keys returned by find(''): {len(find_empty_keys)}")
print(f"Missing from find(None): {len(all_keys - find_none_keys)}")

missing = all_keys - find_none_keys
print("\nSample missing keys:")
for key in list(missing)[:10]:
    print(f"  '{key}'")
```

Output:
```
Total keys in physical_constants: 445
Keys returned by find(None): 355
Keys returned by find(''): 355
Missing from find(None): 90

Sample missing keys:
  'atomic unit of magn. dipole moment'
  'electron-proton magn. moment ratio'
  'tau Compton wavelength over 2 pi'
  'Planck constant over 2 pi'
  'deuteron magn. moment to Bohr magneton ratio'
  'shielded proton magn. moment to nuclear magneton ratio'
  'shielded helion magn. moment to nuclear magneton ratio'
  'atomic unit of 1st hyperpolarizablity'
  'neutron to shielded proton magn. moment ratio'
  'atomic unit of magn. flux density'
```

## Why This Is A Bug

The docstring for `find()` states:

> "Return list of physical_constant keys containing a given string."
>
> "sub : str - Sub-string to search keys for. By default, return all keys."

This documentation clearly states that the function should return keys from `physical_constants`, and when `sub=None` (the default), it should return **all** keys. However, the implementation searches `_current_constants` instead of `physical_constants`, which only contains constants from the current CODATA version (2022), excluding 90 older constants that are still present in `physical_constants`.

This creates an inconsistency where users can access constants via `physical_constants[key]` but cannot find them via `find()`, violating the principle of least surprise.

## Fix

```diff
--- a/scipy/constants/_codata.py
+++ b/scipy/constants/_codata.py
@@ -XX,9 +XX,9 @@ def find(sub: str | None = None, disp: bool = False) -> Any:

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