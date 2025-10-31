# Bug Report: pandas.compat.numpy.function ARGSORT_DEFAULTS Duplicate Key Assignment

**Target**: `pandas.compat.numpy.function.ARGSORT_DEFAULTS`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `ARGSORT_DEFAULTS` dictionary in `pandas/compat/numpy/function.py` assigns the `"kind"` key twice (lines 138 and 140), causing the second assignment (`None`) to overwrite the intended default value (`"quicksort"`).

## Property-Based Test

```python
from pandas.compat.numpy.function import ARGSORT_DEFAULTS

def test_argsort_defaults_no_duplicate_keys():
    assert ARGSORT_DEFAULTS["kind"] == "quicksort", \
        f"Expected kind='quicksort', got kind={ARGSORT_DEFAULTS['kind']!r}"
```

**Failing input**: Not applicable (static configuration bug)

## Reproducing the Bug

```python
from pandas.compat.numpy.function import ARGSORT_DEFAULTS

print(ARGSORT_DEFAULTS)

print(f"\nActual 'kind' value: {ARGSORT_DEFAULTS['kind']!r}")
print(f"Expected: 'quicksort'")
```

## Why This Is A Bug

In `pandas/compat/numpy/function.py` lines 136-141, the code sets:
```python
ARGSORT_DEFAULTS["kind"] = "quicksort"  # Line 138
ARGSORT_DEFAULTS["order"] = None        # Line 139
ARGSORT_DEFAULTS["kind"] = None          # Line 140 - overwrites line 138!
```

This causes `ARGSORT_DEFAULTS["kind"]` to be `None` instead of `"quicksort"`, which doesn't match numpy's actual default parameter value. This affects validation of numpy compatibility parameters in pandas, potentially allowing invalid parameters to pass validation or rejecting valid ones.

## Fix

```diff
--- a/pandas/compat/numpy/function.py
+++ b/pandas/compat/numpy/function.py
@@ -136,8 +136,7 @@ validate_argmax = CompatValidator(
 ARGSORT_DEFAULTS: dict[str, int | str | None] = {}
 ARGSORT_DEFAULTS["axis"] = -1
 ARGSORT_DEFAULTS["kind"] = "quicksort"
 ARGSORT_DEFAULTS["order"] = None
-ARGSORT_DEFAULTS["kind"] = None
 ARGSORT_DEFAULTS["stable"] = None
```