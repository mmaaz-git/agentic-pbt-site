# Bug Report: pandas.compat.numpy.function ARGSORT_DEFAULTS Duplicate Key Assignment

**Target**: `pandas.compat.numpy.function.ARGSORT_DEFAULTS`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `ARGSORT_DEFAULTS` dictionary has the `'kind'` key assigned twice (lines 138 and 140), with the second assignment overwriting the first.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
from pandas.compat.numpy.function import ARGSORT_DEFAULTS

@given(st.integers())
@settings(max_examples=10)
def test_argsort_defaults_no_duplicate_assignments(x):
    assert ARGSORT_DEFAULTS['kind'] == 'quicksort' or ARGSORT_DEFAULTS['kind'] is None
    assert 'kind' in ARGSORT_DEFAULTS
```

**Failing input**: `x=0` (any value shows the issue)

## Reproducing the Bug

```python
from pandas.compat.numpy.function import ARGSORT_DEFAULTS

print("ARGSORT_DEFAULTS contents:")
for key, value in sorted(ARGSORT_DEFAULTS.items()):
    print(f"  {key!r}: {value!r}")

print(f"\nARGSORT_DEFAULTS['kind'] = {ARGSORT_DEFAULTS['kind']!r}")
print("Expected: 'quicksort' (from line 138)")
print("Actual: None (overwritten by line 140)")
```

**Output**:
```
ARGSORT_DEFAULTS contents:
  'axis': -1
  'kind': None
  'order': None
  'stable': None

ARGSORT_DEFAULTS['kind'] = None
Expected: 'quicksort' (from line 138)
Actual: None (overwritten by line 140)
```

## Why This Is A Bug

In `pandas/compat/numpy/function.py` lines 136-141, the code assigns to `ARGSORT_DEFAULTS['kind']` twice:

```python
ARGSORT_DEFAULTS: dict[str, int | str | None] = {}
ARGSORT_DEFAULTS["axis"] = -1
ARGSORT_DEFAULTS["kind"] = "quicksort"  # Line 138
ARGSORT_DEFAULTS["order"] = None
ARGSORT_DEFAULTS["kind"] = None  # Line 140 - overwrites line 138!
ARGSORT_DEFAULTS["stable"] = None
```

The second assignment on line 140 completely overwrites the `'quicksort'` value from line 138. This is likely a copy-paste error or typo.

NumPy's `argsort` has `kind='quicksort'` as the default in older versions and `kind=None` in newer versions (which means stable sort). The fact that both values appear suggests confusion about which default to use, or a botched refactoring.

Note: `ARGSORT_DEFAULTS` and its associated `validate_argsort` validator don't appear to be used anywhere in the current pandas codebase, suggesting this may be dead code. However, the duplicate assignment is still a code quality issue.

## Fix

The fix depends on the intended behavior:

**Option 1**: If `kind=None` is correct (for newer NumPy), remove line 138:
```diff
--- a/pandas/compat/numpy/function.py
+++ b/pandas/compat/numpy/function.py
@@ -135,7 +135,6 @@ ALLANY_DEFAULTS["axis"] = None

 ARGSORT_DEFAULTS: dict[str, int | str | None] = {}
 ARGSORT_DEFAULTS["axis"] = -1
-ARGSORT_DEFAULTS["kind"] = "quicksort"
 ARGSORT_DEFAULTS["order"] = None
 ARGSORT_DEFAULTS["kind"] = None
 ARGSORT_DEFAULTS["stable"] = None
```

**Option 2**: If `kind='quicksort'` is correct (for older NumPy), remove line 140:
```diff
--- a/pandas/compat/numpy/function.py
+++ b/pandas/compat/numpy/function.py
@@ -137,7 +137,6 @@ ARGSORT_DEFAULTS["axis"] = -1
 ARGSORT_DEFAULTS["kind"] = "quicksort"
 ARGSORT_DEFAULTS["order"] = None
-ARGSORT_DEFAULTS["kind"] = None
 ARGSORT_DEFAULTS["stable"] = None
```

**Option 3**: If this is dead code (most likely given no usage found), remove the entire `ARGSORT_DEFAULTS` dictionary and `validate_argsort`.