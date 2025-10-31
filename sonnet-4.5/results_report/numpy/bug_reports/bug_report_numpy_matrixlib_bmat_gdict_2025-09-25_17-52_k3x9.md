# Bug Report: numpy.matrixlib.bmat gdict parameter

**Target**: `numpy.matrixlib.defmatrix.bmat`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When calling `np.bmat()` with a string and providing only the `gdict` parameter (without `ldict`), the function crashes with a `TypeError` instead of correctly looking up variables in the provided global dictionary.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
import warnings

@given(
    st.integers(min_value=1, max_value=5),
    st.integers(min_value=1, max_value=5)
)
def test_bmat_string_with_globals(rows, cols):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PendingDeprecationWarning)

        globals_dict = {
            'A': np.matrix(np.ones((rows, cols))),
            'B': np.matrix(np.zeros((rows, cols)))
        }

        result = np.bmat('A, B', gdict=globals_dict)

        assert result.shape == (rows, 2 * cols)
```

**Failing input**: `rows=1, cols=1` (or any valid values)

## Reproducing the Bug

```python
import warnings
import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore", PendingDeprecationWarning)

    A = np.matrix([[1, 2]])
    result = np.bmat('A', gdict={'A': A})
```

**Output:**
```
TypeError: 'NoneType' object is not subscriptable
```

## Why This Is A Bug

According to the docstring, `bmat` accepts both `gdict` and `ldict` parameters, with `ldict` being optional. When only `gdict` is provided, `ldict` defaults to `None`. However, the internal function `_from_string` attempts to subscript `ldict` directly without checking if it's `None`, causing a `TypeError` instead of the expected `KeyError` that would trigger the fallback to `gdict`.

The issue is in `defmatrix.py` at lines 1029-1035:

```python
try:
    thismat = ldict[col]  # This raises TypeError when ldict is None
except KeyError:
    try:
        thismat = gdict[col]
    except KeyError as e:
        raise NameError(f"name {col!r} is not defined") from None
```

When `ldict` is `None`, the expression `ldict[col]` raises `TypeError` rather than `KeyError`, so the exception handler never catches it and the fallback to `gdict` never happens.

## Fix

```diff
--- a/numpy/matrixlib/defmatrix.py
+++ b/numpy/matrixlib/defmatrix.py
@@ -1027,7 +1027,7 @@ def _from_string(str, gdict, ldict):
         coltup = []
         for col in trow:
             col = col.strip()
-            try:
+            if ldict is not None:
+                try:
+                    thismat = ldict[col]
+                except KeyError:
+                    pass
+            if 'thismat' not in locals():
+                try:
                     thismat = gdict[col]
                 except KeyError as e:
                     raise NameError(f"name {col!r} is not defined") from None
-            except KeyError:
-                try:
-                    thismat = gdict[col]
-                except KeyError as e:
-                    raise NameError(f"name {col!r} is not defined") from None
```

Or more simply:

```diff
--- a/numpy/matrixlib/defmatrix.py
+++ b/numpy/matrixlib/defmatrix.py
@@ -1027,7 +1027,7 @@ def _from_string(str, gdict, ldict):
         coltup = []
         for col in trow:
             col = col.strip()
             try:
-                thismat = ldict[col]
-            except KeyError:
+                thismat = ldict[col] if ldict is not None else gdict[col]
+            except (KeyError, TypeError):
                 try:
                     thismat = gdict[col]
                 except KeyError as e:
                     raise NameError(f"name {col!r} is not defined") from None
```

The simplest fix is to catch `TypeError` in addition to `KeyError` in the exception handler.