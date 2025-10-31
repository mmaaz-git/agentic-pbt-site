# Bug Report: pandas.compat.numpy ARGSORT_DEFAULTS Duplicate Key Assignment

**Target**: `pandas.compat.numpy.function.ARGSORT_DEFAULTS`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `ARGSORT_DEFAULTS` dictionary in `pandas/compat/numpy/function.py` has a duplicate key assignment where `"kind"` is first set to `"quicksort"` (line 138) and then immediately overwritten with `None` (line 140), causing the initial value to be silently discarded.

## Property-Based Test

```python
from pandas.compat.numpy.function import ARGSORT_DEFAULTS

def test_argsort_defaults_duplicate_key():
    assert "kind" in ARGSORT_DEFAULTS
    assert ARGSORT_DEFAULTS["kind"] is None
```

**Observation**: The test reveals that `ARGSORT_DEFAULTS["kind"]` is `None`, not `"quicksort"` as initially assigned.

## Reproducing the Bug

```python
from pandas.compat.numpy.function import ARGSORT_DEFAULTS

print("Current value:", ARGSORT_DEFAULTS["kind"])
print("Expected: 'quicksort' or None (but not both assignments)")
```

## Why This Is A Bug

In `pandas/compat/numpy/function.py` lines 136-141:

```python
ARGSORT_DEFAULTS: dict[str, int | str | None] = {}
ARGSORT_DEFAULTS["axis"] = -1
ARGSORT_DEFAULTS["kind"] = "quicksort"  # Line 138
ARGSORT_DEFAULTS["order"] = None
ARGSORT_DEFAULTS["kind"] = None          # Line 140 - overwrites line 138!
ARGSORT_DEFAULTS["stable"] = None
```

The second assignment on line 140 silently overwrites the first assignment on line 138. This violates the principle of least surprise and suggests either:
1. A copy-paste error where line 140 should set a different key
2. Line 138 should be removed
3. Line 140 should be removed

The presence of `ARGSORT_DEFAULTS_KIND` (lines 150-153) which intentionally omits the `"kind"` key suggests that `ARGSORT_DEFAULTS` was meant to include it with value `"quicksort"`, making line 140 the erroneous one.

## Fix

Remove the duplicate assignment on line 140:

```diff
--- a/pandas/compat/numpy/function.py
+++ b/pandas/compat/numpy/function.py
@@ -137,7 +137,6 @@ ARGSORT_DEFAULTS: dict[str, int | str | None] = {}
 ARGSORT_DEFAULTS["axis"] = -1
 ARGSORT_DEFAULTS["kind"] = "quicksort"
 ARGSORT_DEFAULTS["order"] = None
-ARGSORT_DEFAULTS["kind"] = None
 ARGSORT_DEFAULTS["stable"] = None
```