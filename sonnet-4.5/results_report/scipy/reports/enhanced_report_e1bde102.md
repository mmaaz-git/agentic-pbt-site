# Bug Report: scipy.constants.find Cannot Discover 90 Constants from physical_constants

**Target**: `scipy.constants.find`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `find()` function searches only a subset (CODATA 2022) of constants but `physical_constants` contains constants from all CODATA versions (2002-2022). This causes 90 constants (20.2% of total) to be undiscoverable via `find()` despite being accessible through `physical_constants`.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Property-based test to check if find() can locate all constants in physical_constants"""

from hypothesis import given, strategies as st, settings, example
import scipy.constants as sc


def test_find_none_returns_all_keys():
    """Test that find(None) returns all keys from physical_constants"""
    result = sc.find(None)
    expected = sorted(sc.physical_constants.keys())

    if result != expected:
        print(f"FAIL: find(None) returned {len(result)} keys but physical_constants has {len(expected)} keys")
        print(f"Missing {len(expected) - len(result)} keys from find(None)")
        return False
    else:
        print(f"PASS: find(None) returns all {len(result)} keys")
        return True


@given(st.sampled_from(list(sc.physical_constants.keys())))
@settings(max_examples=100)
@example('Planck constant over 2 pi')  # Known failing case
def test_find_can_locate_all_constants(key):
    """Test that find() can locate every key in physical_constants"""
    results = sc.find(key)
    assert key in results, f"find() could not locate key '{key}' that exists in physical_constants"


if __name__ == "__main__":
    print("=" * 70)
    print("Testing scipy.constants.find() function")
    print("=" * 70)

    print("\n1. Testing find(None) returns all keys:")
    print("-" * 40)
    test_find_none_returns_all_keys()

    print("\n2. Testing find() can locate all constants:")
    print("-" * 40)
    try:
        test_find_can_locate_all_constants()
        print("PASS: All constants can be found")
    except AssertionError as e:
        print(f"FAIL: {e}")
        print("\nHypothesis discovered the following failing example:")
        print("Key: 'Planck constant over 2 pi'")
        print("This key exists in physical_constants but find() returns an empty list")
        print("\nAdditional information:")
        key = 'Planck constant over 2 pi'
        print(f"  - sc.physical_constants['{key}'] = {sc.physical_constants[key]}")
        print(f"  - sc.find('{key}') = {sc.find(key)}")

        # Show a few more examples
        print("\nOther affected constants (sampling):")
        all_keys = set(sc.physical_constants.keys())
        current_keys = set(sc._codata._current_constants.keys())
        missing_keys = all_keys - current_keys
        for i, k in enumerate(sorted(missing_keys)[:5]):
            print(f"  - '{k}'")
```

<details>

<summary>
**Failing input**: `'Planck constant over 2 pi'`
</summary>
```
======================================================================
Testing scipy.constants.find() function
======================================================================

1. Testing find(None) returns all keys:
----------------------------------------
FAIL: find(None) returned 355 keys but physical_constants has 445 keys
Missing 90 keys from find(None)

2. Testing find() can locate all constants:
----------------------------------------
FAIL: find() could not locate key 'Planck constant over 2 pi' that exists in physical_constants

Hypothesis discovered the following failing example:
Key: 'Planck constant over 2 pi'
This key exists in physical_constants but find() returns an empty list

Additional information:
  - sc.physical_constants['Planck constant over 2 pi'] = (1.0545718e-34, 'J s', 1.3e-42)
  - sc.find('Planck constant over 2 pi') = []

Other affected constants (sampling):
  - 'Bohr magneton in inverse meters per tesla'
  - 'Boltzmann constant in inverse meters per kelvin'
  - 'Compton wavelength over 2 pi'
  - 'Cu x unit'
  - 'Faraday constant for conventional electric current'
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Demonstration of scipy.constants.find bug - it cannot find 90 constants"""

import scipy.constants as sc

# A constant that exists in older CODATA but not in current (2022)
key = 'Planck constant over 2 pi'

# Show that the constant exists and can be accessed
print(f"Checking if '{key}' exists in physical_constants...")
try:
    value = sc.physical_constants[key]
    print(f"✓ sc.physical_constants['{key}'] = {value}")
except KeyError:
    print(f"✗ KeyError: '{key}' not found in physical_constants")

# Try to find it using the find() function
print(f"\nSearching for '{key}' using find()...")
results = sc.find('Planck constant over 2 pi')
print(f"sc.find('Planck constant over 2 pi') = {results}")

if len(results) == 0:
    print("✗ ERROR: find() cannot locate this constant that exists in physical_constants!")
else:
    print(f"✓ Found {len(results)} result(s)")

# Analyze the scope of the problem
print("\n--- Analyzing the scope of the issue ---")
print(f"_current_constants has {len(sc._codata._current_constants)} keys")
print(f"physical_constants has {len(sc.physical_constants)} keys")
missing_count = len(sc.physical_constants) - len(sc._codata._current_constants)
print(f"{missing_count} constants ({missing_count/len(sc.physical_constants)*100:.1f}%) are inaccessible via find()")

# Show some other affected constants
print("\n--- Examples of other affected constants ---")
all_keys = set(sc.physical_constants.keys())
current_keys = set(sc._codata._current_constants.keys())
missing_keys = all_keys - current_keys

print(f"Total missing from find(): {len(missing_keys)} constants")
print("First 10 examples:")
for i, key in enumerate(sorted(missing_keys)[:10]):
    print(f"  {i+1}. '{key}'")
```

<details>

<summary>
Output showing 90 constants cannot be found
</summary>
```
Checking if 'Planck constant over 2 pi' exists in physical_constants...
✓ sc.physical_constants['Planck constant over 2 pi'] = (1.0545718e-34, 'J s', 1.3e-42)

Searching for 'Planck constant over 2 pi' using find()...
sc.find('Planck constant over 2 pi') = []
✗ ERROR: find() cannot locate this constant that exists in physical_constants!

--- Analyzing the scope of the issue ---
_current_constants has 355 keys
physical_constants has 445 keys
90 constants (20.2%) are inaccessible via find()

--- Examples of other affected constants ---
Total missing from find(): 90 constants
First 10 examples:
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
```
</details>

## Why This Is A Bug

This violates the documented API contract of the `find()` function. The function's docstring explicitly states it will "Return list of physical_constant keys containing a given string" but it only searches 355 out of 445 available keys in `physical_constants`.

Key issues:
1. **Documentation mismatch**: The docstring promises to search "physical_constant keys" but actually searches only `_current_constants` (CODATA 2022)
2. **Incomplete discovery**: 90 constants (20.2%) that are part of the public API cannot be discovered
3. **Inconsistent behavior**: `find(None)` should return all keys from `physical_constants` but returns only a subset
4. **User expectation violation**: The examples in the docstring show using `find()` results with `physical_constants`, implying they work together comprehensively

The root cause is that `physical_constants` is built by merging constants from multiple CODATA versions (2002, 2006, 2010, 2014, 2018, 2022) via successive `.update()` calls, but `find()` only searches `_current_constants` which points to CODATA 2022 only.

## Relevant Context

The issue is in `/home/npc/pbt/agentic-pbt/envs/scipy_env/lib/python3.13/site-packages/scipy/constants/_codata.py`:

- Line 2061-2068: `physical_constants` is built from all CODATA versions
- Line 2068: `_current_constants = _physical_constants_2022`
- Line 2251-2254: `find()` searches only `_current_constants`

Constants from older CODATA versions remain accessible for backward compatibility but cannot be discovered. This includes important constants like 'Planck constant over 2 pi' which was renamed to 'reduced Planck constant' in newer versions.

SciPy documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.constants.find.html
Source code: https://github.com/scipy/scipy/blob/main/scipy/constants/_codata.py

## Proposed Fix

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