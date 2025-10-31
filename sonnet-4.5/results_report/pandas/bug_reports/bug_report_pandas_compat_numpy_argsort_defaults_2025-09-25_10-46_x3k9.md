# Bug Report: pandas.compat.numpy ARGSORT_DEFAULTS Duplicate Assignment

**Target**: `pandas.compat.numpy.function.ARGSORT_DEFAULTS`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `ARGSORT_DEFAULTS` dictionary in `pandas/compat/numpy/function.py` has a duplicate assignment for the `"kind"` key, where line 137 sets it to `"quicksort"` and line 139 immediately overwrites it to `None`, making line 137 dead code.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.compat.numpy.function import ARGSORT_DEFAULTS


@given(st.just(None))
def test_argsort_defaults_duplicate_assignment(expected):
    actual = ARGSORT_DEFAULTS['kind']
    assert actual == expected
```

**Failing input**: N/A (the bug is in the source code itself, not triggered by specific inputs)

## Reproducing the Bug

```python
from pandas.compat.numpy.function import ARGSORT_DEFAULTS

print("ARGSORT_DEFAULTS:", ARGSORT_DEFAULTS)
print("ARGSORT_DEFAULTS['kind']:", ARGSORT_DEFAULTS['kind'])

expected_from_line_137 = "quicksort"
actual_value = ARGSORT_DEFAULTS['kind']

print(f"\nLine 137 sets kind='{expected_from_line_137}'")
print(f"Line 139 overwrites kind={actual_value}")
print(f"Result: Line 137 has no effect (dead code)")
```

Output:
```
ARGSORT_DEFAULTS: {'axis': -1, 'kind': None, 'order': None, 'stable': None}
ARGSORT_DEFAULTS['kind']: None

Line 137 sets kind='quicksort'
Line 139 overwrites kind=None
Result: Line 137 has no effect (dead code)
```

## Why This Is A Bug

Lines 135-140 in `function.py` contain:

```python
ARGSORT_DEFAULTS: dict[str, int | str | None] = {}
ARGSORT_DEFAULTS["axis"] = -1
ARGSORT_DEFAULTS["kind"] = "quicksort"  # Line 137
ARGSORT_DEFAULTS["order"] = None
ARGSORT_DEFAULTS["kind"] = None         # Line 139 - overwrites line 137
ARGSORT_DEFAULTS["stable"] = None
```

The assignment at line 137 is immediately overwritten by line 139, making it dead code. This is confusing for maintainers and suggests either:
1. Line 137 is leftover code that should be deleted
2. Line 139 was added by mistake
3. Copy-paste error

While this doesn't cause runtime errors (the final value `None` is correct per NumPy 1.24+ defaults), it creates code confusion and maintenance burden.

## Fix

Remove the dead code at line 137:

```diff
 ARGSORT_DEFAULTS: dict[str, int | str | None] = {}
 ARGSORT_DEFAULTS["axis"] = -1
-ARGSORT_DEFAULTS["kind"] = "quicksort"
 ARGSORT_DEFAULTS["order"] = None
 ARGSORT_DEFAULTS["kind"] = None
 ARGSORT_DEFAULTS["stable"] = None
```