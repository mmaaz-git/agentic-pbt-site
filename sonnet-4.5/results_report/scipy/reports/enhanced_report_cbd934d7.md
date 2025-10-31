# Bug Report: scipy.constants.find() Returns Incomplete Key List

**Target**: `scipy.constants.find()`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `find()` function violates its documented contract by returning only 355 keys from the current CODATA 2022 dataset instead of all 445 keys from `physical_constants`, missing 90 constants (20% of total) when called with `sub=None` or `sub=''`.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
import scipy.constants as sc

def test_find_none_returns_all_keys():
    """Test that find(None) returns all keys from physical_constants."""
    results = sc.find(None)
    expected = set(sc.physical_constants.keys())
    actual = set(results)

    if actual != expected:
        missing = expected - actual
        extra = actual - expected
        print(f"FAILED: find(None) does not return all physical_constants keys")
        print(f"  Expected {len(expected)} keys, got {len(actual)} keys")
        if missing:
            print(f"  Missing {len(missing)} keys from physical_constants")
            print(f"    First 5 missing: {list(missing)[:5]}")
        if extra:
            print(f"  Extra {len(extra)} keys not in physical_constants")
            print(f"    First 5 extra: {list(extra)[:5]}")
        assert False, f"find(None) returned {len(actual)} keys but physical_constants has {len(expected)} keys"
    else:
        print("PASSED: find(None) returns all physical_constants keys")

def test_find_empty_string_returns_all_keys():
    """Test that find('') returns all keys from physical_constants."""
    results = sc.find('')
    expected = set(sc.physical_constants.keys())
    actual = set(results)

    if actual != expected:
        missing = expected - actual
        extra = actual - expected
        print(f"FAILED: find('') does not return all physical_constants keys")
        print(f"  Expected {len(expected)} keys, got {len(actual)} keys")
        if missing:
            print(f"  Missing {len(missing)} keys from physical_constants")
            print(f"    First 5 missing: {list(missing)[:5]}")
        if extra:
            print(f"  Extra {len(extra)} keys not in physical_constants")
            print(f"    First 5 extra: {list(extra)[:5]}")
        assert False, f"find('') returned {len(actual)} keys but physical_constants has {len(expected)} keys"
    else:
        print("PASSED: find('') returns all physical_constants keys")

if __name__ == "__main__":
    print("Running property-based tests for scipy.constants.find()\n")
    print("=" * 60)

    try:
        test_find_none_returns_all_keys()
    except AssertionError as e:
        print(f"AssertionError: {e}")

    print("\n" + "=" * 60)

    try:
        test_find_empty_string_returns_all_keys()
    except AssertionError as e:
        print(f"AssertionError: {e}")
```

<details>

<summary>
**Failing input**: `find(None)` and `find('')`
</summary>
```
Running property-based tests for scipy.constants.find()

============================================================
FAILED: find(None) does not return all physical_constants keys
  Expected 445 keys, got 355 keys
  Missing 90 keys from physical_constants
    First 5 missing: ['nuclear magneton in inverse meters per tesla', 'shielded helion to shielded proton magn. moment ratio', 'Planck constant over 2 pi times c in MeV fm', 'Boltzmann constant in inverse meters per kelvin', 'electron-neutron magn. moment ratio']
AssertionError: find(None) returned 355 keys but physical_constants has 445 keys

============================================================
FAILED: find('') does not return all physical_constants keys
  Expected 445 keys, got 355 keys
  Missing 90 keys from physical_constants
    First 5 missing: ['nuclear magneton in inverse meters per tesla', 'shielded helion to shielded proton magn. moment ratio', 'Planck constant over 2 pi times c in MeV fm', 'Boltzmann constant in inverse meters per kelvin', 'electron-neutron magn. moment ratio']
AssertionError: find('') returned 355 keys but physical_constants has 445 keys
```
</details>

## Reproducing the Bug

```python
import scipy.constants as sc

# Get all the keys
all_keys = set(sc.physical_constants.keys())
find_none_keys = set(sc.find(None))
find_empty_keys = set(sc.find(''))

# Print statistics
print(f"Total keys in physical_constants: {len(all_keys)}")
print(f"Keys returned by find(None): {len(find_none_keys)}")
print(f"Keys returned by find(''): {len(find_empty_keys)}")
print(f"Missing from find(None): {len(all_keys - find_none_keys)}")
print(f"Missing from find(''): {len(all_keys - find_empty_keys)}")

# Show some missing keys
missing = all_keys - find_none_keys
print(f"\nSample of missing keys (showing first 10):")
for i, key in enumerate(sorted(missing)[:10]):
    print(f"  {i+1}. '{key}'")

# Verify these missing keys are actually accessible
print("\nVerifying missing keys are accessible:")
sample_key = list(missing)[0]
print(f"  Accessing '{sample_key}':")
print(f"    Value: {sc.physical_constants[sample_key]}")
```

<details>

<summary>
Output showing 90 missing keys from find() that exist in physical_constants
</summary>
```
Total keys in physical_constants: 445
Keys returned by find(None): 355
Keys returned by find(''): 355
Missing from find(None): 90
Missing from find(''): 90

Sample of missing keys (showing first 10):
  1. 'Bohr magneton in inverse meters per tesla'
  2. 'Boltzmann constant in inverse meters per kelvin'
  3. 'Compton wavelength over 2 pi'
  4. 'Cu x unit'
  5. 'Faraday constant for conventional electric current'
  6. 'Mo x unit'
  7. 'Planck constant in eV s'
  8. 'Planck constant over 2 pi'
  9. 'Planck constant over 2 pi in eV s'
  10. 'Planck constant over 2 pi times c in MeV fm'

Verifying missing keys are accessible:
  Accessing 'deuteron magn. moment to nuclear magneton ratio':
    Value: (0.8574382329, '', 9.2e-09)
```
</details>

## Why This Is A Bug

The `scipy.constants.find()` function explicitly violates its documented contract in multiple ways:

1. **Documentation states**: "Return list of physical_constant keys containing a given string" - The function name and docstring clearly indicate it should search `physical_constants`, not a subset.

2. **Default behavior claim**: The docstring parameter description states "By default, return all keys" when `sub=None`. This is unambiguous - it should return ALL keys, not just current CODATA keys.

3. **Inconsistency with module design**: The `physical_constants` dictionary is the public API that contains 445 constants from multiple CODATA versions (2002, 2006, 2010, 2014, 2018, 2022). The `find()` function only searches `_current_constants` (CODATA 2022 with 355 keys), which is an internal implementation detail.

4. **Violates discoverability principle**: Users can access constants like `physical_constants['Planck constant over 2 pi']` directly, but `find('Planck constant over 2 pi')` returns an empty list, making these constants undiscoverable through the search interface.

5. **No documentation of limitation**: Neither the docstring nor the official SciPy documentation mentions that `find()` only searches the current CODATA version or that some keys in `physical_constants` won't be found.

## Relevant Context

The `scipy.constants` module maintains physical constants from multiple CODATA releases for backwards compatibility and reproducibility. The implementation details show:

- `physical_constants` is built by merging all CODATA versions (2002-2022)
- `_current_constants` only contains CODATA 2022 constants
- The `find()` function searches `_current_constants` instead of `physical_constants`
- 90 constants from older CODATA versions are accessible but not findable

This is particularly problematic for:
- Users needing historical constants for reproducing older calculations
- Scientists comparing values across CODATA versions
- Anyone trying to discover available constants programmatically

The implementation is located in `/scipy/constants/_codata.py` where the problematic code searches `_current_constants` instead of the documented `physical_constants`.

## Proposed Fix

```diff
--- a/scipy/constants/_codata.py
+++ b/scipy/constants/_codata.py
@@ -XXX,9 +XXX,9 @@ def find(sub: str | None = None, disp: bool = False) -> Any:

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