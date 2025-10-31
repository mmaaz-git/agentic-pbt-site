# Bug Report: pandas.compat.numpy ARGSORT_DEFAULTS Inconsistency

**Target**: `pandas.compat.numpy.function.ARGSORT_DEFAULTS`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

ARGSORT_DEFAULTS contains dead code and an inconsistency with SORT_DEFAULTS for the 'kind' parameter, despite both tracking the same numpy default value.

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
        "ARGSORT_DEFAULTS and SORT_DEFAULTS should have the same 'kind' default"
```

**Failing input**: N/A (configuration bug)

## Reproducing the Bug

```python
from pandas.compat.numpy.function import ARGSORT_DEFAULTS, SORT_DEFAULTS

print(f"ARGSORT_DEFAULTS['kind'] = {ARGSORT_DEFAULTS['kind']!r}")
print(f"SORT_DEFAULTS['kind'] = {SORT_DEFAULTS['kind']!r}")

assert ARGSORT_DEFAULTS['kind'] != SORT_DEFAULTS['kind']
```

Output:
```
ARGSORT_DEFAULTS['kind'] = None
SORT_DEFAULTS['kind'] = 'quicksort'
```

## Why This Is A Bug

The file's purpose (per docstring at top) is to provide "commonly used default arguments" to "make it easier to adjust to future upstream changes in the analogous numpy signatures." Both numpy.sort and numpy.argsort have identical 'kind' parameter defaults (None in signature, quicksort in behavior), yet pandas tracks them inconsistently.

Additionally, line 138 sets `ARGSORT_DEFAULTS["kind"] = "quicksort"` which is immediately overwritten by line 140 setting it to `None`. This dead code suggests unintentional behavior.

While both values work correctly (numpy treats None as quicksort), the inconsistency:
1. Defeats the module's purpose of centralized default tracking
2. Contains confusing dead code
3. Makes future numpy compatibility updates error-prone

## Fix

```diff
--- a/pandas/compat/numpy/function.py
+++ b/pandas/compat/numpy/function.py
@@ -135,10 +135,8 @@ def validate_argmax_with_skipna(skipna: bool | ndarray | None, args, kwargs) -

 ARGSORT_DEFAULTS: dict[str, int | str | None] = {}
 ARGSORT_DEFAULTS["axis"] = -1
-ARGSORT_DEFAULTS["kind"] = "quicksort"
 ARGSORT_DEFAULTS["order"] = None
 ARGSORT_DEFAULTS["kind"] = None
 ARGSORT_DEFAULTS["stable"] = None


 validate_argsort = CompatValidator(
```