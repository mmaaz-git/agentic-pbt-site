# Bug Report: pandas.compat.numpy.function Dead Code and ARGSORT_DEFAULTS Inconsistency

**Target**: `pandas.compat.numpy.function.ARGSORT_DEFAULTS`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

ARGSORT_DEFAULTS contains dead code (line 138 immediately overwritten by line 140) and tracks the 'kind' parameter inconsistently with SORT_DEFAULTS, despite both numpy.sort and numpy.argsort having identical default values.

## Property-Based Test

```python
import pytest
from pandas.compat.numpy.function import ARGSORT_DEFAULTS, SORT_DEFAULTS


def test_argsort_sort_defaults_consistency():
    """
    Property: ARGSORT_DEFAULTS and SORT_DEFAULTS should have consistent 'kind' values
    since both numpy.sort and numpy.argsort have the same default sorting algorithm.
    """
    assert ARGSORT_DEFAULTS['kind'] == SORT_DEFAULTS['kind'], \
        f"ARGSORT_DEFAULTS and SORT_DEFAULTS should have the same 'kind' default, but got {ARGSORT_DEFAULTS['kind']!r} != {SORT_DEFAULTS['kind']!r}"

if __name__ == "__main__":
    test_argsort_sort_defaults_consistency()
```

<details>

<summary>
**Failing input**: N/A (configuration inconsistency, not input-based)
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 14, in <module>
    test_argsort_sort_defaults_consistency()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 10, in test_argsort_sort_defaults_consistency
    assert ARGSORT_DEFAULTS['kind'] == SORT_DEFAULTS['kind'], \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: ARGSORT_DEFAULTS and SORT_DEFAULTS should have the same 'kind' default, but got None != 'quicksort'
```
</details>

## Reproducing the Bug

```python
from pandas.compat.numpy.function import ARGSORT_DEFAULTS, SORT_DEFAULTS

print(f"ARGSORT_DEFAULTS['kind'] = {ARGSORT_DEFAULTS['kind']!r}")
print(f"SORT_DEFAULTS['kind'] = {SORT_DEFAULTS['kind']!r}")

# Demonstrate the inconsistency
print(f"\nInconsistency detected: ARGSORT_DEFAULTS['kind'] != SORT_DEFAULTS['kind']")
print(f"    ARGSORT_DEFAULTS['kind'] = {ARGSORT_DEFAULTS['kind']!r}")
print(f"    SORT_DEFAULTS['kind'] = {SORT_DEFAULTS['kind']!r}")

# Check if both numpy functions have the same default
import numpy as np
import inspect

sort_sig = inspect.signature(np.sort)
argsort_sig = inspect.signature(np.argsort)

print(f"\nnumpy.sort default 'kind' parameter: {sort_sig.parameters['kind'].default!r}")
print(f"numpy.argsort default 'kind' parameter: {argsort_sig.parameters['kind'].default!r}")

print("\nBoth numpy functions have the same 'kind' default (None), but pandas tracks them inconsistently.")

# The assertion that would fail
assert ARGSORT_DEFAULTS['kind'] != SORT_DEFAULTS['kind'], "The inconsistency is confirmed"
```

<details>

<summary>
Demonstration of inconsistent default tracking
</summary>
```
ARGSORT_DEFAULTS['kind'] = None
SORT_DEFAULTS['kind'] = 'quicksort'

Inconsistency detected: ARGSORT_DEFAULTS['kind'] != SORT_DEFAULTS['kind']
    ARGSORT_DEFAULTS['kind'] = None
    SORT_DEFAULTS['kind'] = 'quicksort'

numpy.sort default 'kind' parameter: None
numpy.argsort default 'kind' parameter: None

Both numpy functions have the same 'kind' default (None), but pandas tracks them inconsistently.
```
</details>

## Why This Is A Bug

This violates the module's documented purpose and contains dead code:

1. **Dead code exists**: Lines 138-140 in `/pandas/compat/numpy/function.py` show:
   ```python
   ARGSORT_DEFAULTS["kind"] = "quicksort"  # Line 138 - immediately overwritten
   ARGSORT_DEFAULTS["order"] = None        # Line 139
   ARGSORT_DEFAULTS["kind"] = None         # Line 140 - overwrites line 138
   ```
   Line 138 is dead code that gets immediately overwritten, indicating unintentional behavior.

2. **Module purpose violated**: The file's docstring (lines 14-16) explicitly states: "This module provides a set of commonly used default arguments... This module will make it easier to adjust to future upstream changes in the analogous numpy signatures." Having different values for identical NumPy defaults contradicts this centralized tracking purpose.

3. **NumPy consistency**: Both `numpy.sort` and `numpy.argsort` have identical 'kind' parameter behavior:
   - Both function signatures use `kind=None`
   - Both treat `None` as 'quicksort' (actually introsort internally)
   - NumPy documentation confirms both have "The default is 'quicksort'"

4. **Maintenance risk**: Future developers updating NumPy compatibility will be confused by why identical defaults are tracked differently, potentially leading to errors when NumPy changes its defaults.

## Relevant Context

The pandas compatibility module (`pandas/compat/numpy/function.py`) provides centralized default argument tracking for NumPy function compatibility. The module uses these defaults in validators that ensure pandas functions properly handle NumPy-style arguments.

Looking at the code structure:
- `SORT_DEFAULTS` (line 279) correctly sets `"kind": "quicksort"`
- `ARGSORT_DEFAULTS` has conflicting assignments on lines 138 and 140
- Both are used by `CompatValidator` instances to validate function arguments

The bug appears to be a copy-paste or merge error where line 138 was intended to be the final value but was accidentally overwritten. Since both `None` and `'quicksort'` work identically in NumPy (None is interpreted as quicksort), this bug doesn't cause runtime errors but creates maintenance confusion.

Relevant NumPy documentation:
- [numpy.sort documentation](https://numpy.org/doc/stable/reference/generated/numpy.sort.html)
- [numpy.argsort documentation](https://numpy.org/doc/stable/reference/generated/numpy.argsort.html)

## Proposed Fix

```diff
--- a/pandas/compat/numpy/function.py
+++ b/pandas/compat/numpy/function.py
@@ -135,10 +135,9 @@ def validate_argmax_with_skipna(skipna: bool | ndarray | None, args, kwargs) -

 ARGSORT_DEFAULTS: dict[str, int | str | None] = {}
 ARGSORT_DEFAULTS["axis"] = -1
-ARGSORT_DEFAULTS["kind"] = "quicksort"
+ARGSORT_DEFAULTS["kind"] = "quicksort"
 ARGSORT_DEFAULTS["order"] = None
-ARGSORT_DEFAULTS["kind"] = None
 ARGSORT_DEFAULTS["stable"] = None


 validate_argsort = CompatValidator(
```