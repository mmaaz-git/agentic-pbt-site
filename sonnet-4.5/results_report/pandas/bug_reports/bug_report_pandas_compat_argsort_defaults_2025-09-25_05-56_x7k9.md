# Bug Report: pandas.compat.numpy.function ARGSORT_DEFAULTS Duplicate Assignment

**Target**: `pandas.compat.numpy.function.ARGSORT_DEFAULTS`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `ARGSORT_DEFAULTS` dictionary has a duplicate assignment for the `"kind"` key on lines 138 and 140, causing the first value (`"quicksort"`) to be immediately overwritten by `None`.

## Property-Based Test

```python
from hypothesis import given, strategies as st

from pandas.compat.numpy.function import ARGSORT_DEFAULTS

@given(st.just(None))
def test_argsort_defaults_kind_value(dummy):
    assert ARGSORT_DEFAULTS["kind"] is None
```

**Failing input**: N/A (static code bug)

## Reproducing the Bug

```python
from pandas.compat.numpy.function import ARGSORT_DEFAULTS

print("ARGSORT_DEFAULTS:", ARGSORT_DEFAULTS)

assert ARGSORT_DEFAULTS["kind"] is None
```

## Why This Is A Bug

Lines 138-140 in `/pandas/compat/numpy/function.py`:
```python
ARGSORT_DEFAULTS["kind"] = "quicksort"
ARGSORT_DEFAULTS["order"] = None
ARGSORT_DEFAULTS["kind"] = None
```

The first assignment (`"kind" = "quicksort"`) is immediately overwritten by the third line (`"kind" = None`), making line 138 dead code. This is likely a typo where line 140 was supposed to assign to a different key or line 138 should be removed.

## Fix

```diff
--- a/pandas/compat/numpy/function.py
+++ b/pandas/compat/numpy/function.py
@@ -135,7 +135,6 @@

 ARGSORT_DEFAULTS: dict[str, int | str | None] = {}
 ARGSORT_DEFAULTS["axis"] = -1
-ARGSORT_DEFAULTS["kind"] = "quicksort"
 ARGSORT_DEFAULTS["order"] = None
 ARGSORT_DEFAULTS["kind"] = None
 ARGSORT_DEFAULTS["stable"] = None
```