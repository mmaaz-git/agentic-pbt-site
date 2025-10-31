# Bug Report: numpy.matrixlib Empty String Parsing Creates Degenerate Matrices

**Target**: `numpy.matrixlib.matrix` (specifically `_convert_from_string`)
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The matrix string parser allows creation of matrices with zero-size dimensions when given empty or whitespace-only strings, violating the invariant that matrices should have positive dimensions.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from numpy import matrix

@given(st.text(alphabet="0123456789 ,;", min_size=1, max_size=50))
def test_string_parsing_creates_valid_matrices(s):
    try:
        m = matrix(s)
        assert m.ndim == 2, f"Matrix must be 2D, got {m.ndim}D"
        assert all(dim > 0 for dim in m.shape), f"Matrix has zero dimension: {m.shape}"
    except (ValueError, SyntaxError):
        pass
```

**Failing input**: `";"`

## Reproducing the Bug

```python
import numpy as np
from numpy import matrix

m = matrix(";")
print(f"Shape: {m.shape}")
print(f"Size: {m.size}")

assert m.shape == (2, 0)
assert m.size == 0
```

## Why This Is A Bug

The semicolon `;` is used to separate rows in matrix string notation. A single semicolon implies two empty rows, which `_convert_from_string` parses into `[[], []]` - a valid Python list representing a 2×0 matrix. However, such degenerate matrices with zero-size dimensions are mathematically questionable and likely not the user's intent. The parser should either reject such inputs with a clear error message or handle them more gracefully.

The bug occurs in `_convert_from_string` (defmatrix.py:16-33):
- For input `";"`, `data.split(';')` returns `['', '']`
- Each empty string row becomes an empty list after parsing
- This creates `[[], []]` which numpy.array converts to shape (2, 0)

Similar issues occur with:
- `""` (empty string)
- `" "` (whitespace only)
- `";;"` (multiple semicolons)

## Fix

```diff
--- a/numpy/matrixlib/defmatrix.py
+++ b/numpy/matrixlib/defmatrix.py
@@ -14,6 +14,9 @@ from numpy.linalg import matrix_power


 def _convert_from_string(data):
+    if not data or not data.strip():
+        raise ValueError("Empty string cannot be converted to a matrix")
+
     for char in '[]':
         data = data.replace(char, '')

@@ -25,6 +28,8 @@ def _convert_from_string(data):
         for col in trow:
             temp = col.split()
             newrow.extend(map(ast.literal_eval, temp))
+        if not newrow:
+            raise ValueError("Empty rows are not allowed in matrix string notation")
         if count == 0:
             Ncols = len(newrow)
         elif len(newrow) != Ncols:
```