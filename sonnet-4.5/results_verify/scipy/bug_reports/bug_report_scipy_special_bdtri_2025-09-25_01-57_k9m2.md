# Bug Report: scipy.special.bdtri Returns NaN Without Error When k >= n

**Target**: `scipy.special.bdtri`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`scipy.special.bdtri` silently returns NaN when k >= n, violating its documented inverse relationship with `bdtr`. The function should either document this constraint, raise an informative error, or handle the edge case gracefully.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
import scipy.special as sp
import numpy as np

@given(
    st.integers(min_value=0, max_value=50),
    st.integers(min_value=1, max_value=50),
    st.floats(min_value=0.01, max_value=0.99)
)
@settings(max_examples=500)
def test_bdtri_round_trip(k, n, p):
    assume(k <= n)

    y = sp.bdtr(k, n, p)
    p_reconstructed = sp.bdtri(k, n, y)
    y_reconstructed = sp.bdtr(k, n, p_reconstructed)

    assert np.isclose(y, y_reconstructed, rtol=1e-8, atol=1e-10)
```

**Failing input**: `k=1, n=1, p=0.5` (or any case where k >= n)

## Reproducing the Bug

```python
import numpy as np
import scipy.special as sp

k, n, p = 1, 1, 0.5

y = sp.bdtr(k, n, p)
print(f"bdtr({k}, {n}, {p}) = {y}")

p_reconstructed = sp.bdtri(k, n, y)
print(f"bdtri({k}, {n}, {y}) = {p_reconstructed}")

y_reconstructed = sp.bdtr(k, n, p_reconstructed)
print(f"bdtr({k}, {n}, {p_reconstructed}) = {y_reconstructed}")

assert np.isclose(y, y_reconstructed)
```

Output:
```
bdtr(1, 1, 0.5) = 1.0
bdtri(1, 1, 1.0) = nan
bdtr(1, 1, nan) = nan
AssertionError
```

## Why This Is A Bug

**Mathematical Context**: When k >= n, `bdtr(k, n, p)` correctly returns 1.0 for any value of p, because we're summing the entire binomial probability mass (outcomes 0 through n). However, this makes the inverse function `bdtri` mathematically ill-defined, as there's no unique p that satisfies the equation.

**API Contract Violation**: The documentation for `bdtri` states:

> "Finds the event probability `p` such that the sum of the terms 0 through `k` of the binomial probability density is equal to the given cumulative probability `y`."

The documentation does not mention that k must be < n, nor does it specify what happens when the inverse is not uniquely defined. Users would reasonably expect either:

1. An informative error message (e.g., `ValueError: k must be less than n for bdtri`)
2. A documented canonical return value (e.g., 0.5 when multiple solutions exist)
3. Clear documentation of the k < n constraint

**Impact**: Users who inadvertently call `bdtri` with k >= n receive NaN with no warning, leading to silent failures in downstream calculations. This is particularly problematic in automated pipelines where NaN propagation can corrupt results.

## Fix

Add input validation to `bdtri` to raise an informative error when k >= n:

```diff
--- a/scipy/special/_cdflib.py
+++ b/scipy/special/_cdflib.py
@@ -bdtri_function_location
+    if k >= n:
+        raise ValueError(
+            f"bdtri requires k < n for a unique inverse. "
+            f"Got k={k}, n={n}. When k >= n, bdtr always returns 1.0, "
+            f"making the inverse non-unique."
+        )
+
     # existing bdtri implementation
```

Alternatively, document this behavior in the docstring:

```diff
--- a/scipy/special/_basic.py
+++ b/scipy/special/_basic.py
@@ -bdtri_docstring_location
 Inverse function to `bdtr` with respect to `p`.

 Finds the event probability `p` such that the sum of the terms 0 through
 `k` of the binomial probability density is equal to the given cumulative
 probability `y`.

+.. note::
+   This function requires k < n. When k >= n, `bdtr` always returns 1.0
+   regardless of p, making the inverse undefined. In such cases, `bdtri`
+   returns NaN.
+
 Parameters
```