# Bug Report: scipy.integrate.newton_cotes List Input Handling

**Target**: `scipy.integrate.newton_cotes`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`newton_cotes` crashes with a TypeError when given a list input with non-uniform spacing and `equal=0`. The function fails to convert the list to a numpy array before performing arithmetic operations, despite other scipy.integrate functions (trapezoid, simpson) correctly handling list inputs.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
from scipy import integrate


@given(
    spacing=st.lists(st.floats(min_value=0.1, max_value=1.9, allow_nan=False, allow_infinity=False), min_size=0, max_size=3)
)
@settings(max_examples=100)
def test_newton_cotes_handles_list_input(spacing):
    spacing_sorted = sorted(set(spacing))
    rn = [0.0] + [s for s in spacing_sorted] + [float(len(spacing_sorted) + 1)]

    an, Bn = integrate.newton_cotes(rn, equal=0)
    assert np.all(np.isfinite(an))
```

**Failing input**: `rn = [0, 0.5, 2]` (or any list with non-uniform spacing)

## Reproducing the Bug

```python
import numpy as np
from scipy import integrate

rn_list = [0, 0.5, 2]

an, Bn = integrate.newton_cotes(rn_list, equal=0)
```

Output:
```
TypeError: unsupported operand type(s) for /: 'list' and 'float'
```

The same call works fine with a numpy array:
```python
rn_array = np.array([0, 0.5, 2])
an, Bn = integrate.newton_cotes(rn_array, equal=0)
```

## Why This Is A Bug

1. **Inconsistent behavior**: Other scipy.integrate functions (trapezoid, simpson, etc.) accept both lists and arrays
2. **Documentation doesn't specify array requirement**: The docstring describes `rn` as "int" or implicitly a sequence, not specifically requiring numpy.ndarray
3. **Partial support is confusing**: The function works with lists when `equal=1` or when spacing is uniform, but crashes with non-uniform spacing
4. **Poor error message**: The TypeError occurs deep in the function without explaining that the input should be an array

## Fix

Add array conversion after determining the input type:

```diff
--- a/scipy/integrate/_quadrature.py
+++ b/scipy/integrate/_quadrature.py
@@ -1044,6 +1044,9 @@ def newton_cotes(rn, equal=0):
         rn = np.arange(N+1)
         equal = 1

+    # Ensure rn is a numpy array for arithmetic operations
+    rn = np.asarray(rn)
+
     if equal and N in _builtincoeffs:
         na, da, vi, nb, db = _builtincoeffs[N]
         an = na * np.array(vi, dtype=float) / da
```

This ensures `rn` is always a numpy array before any arithmetic operations at line 1056 (`yi = rn / float(N)`).