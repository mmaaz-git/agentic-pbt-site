# Bug Report: numpy.random.dirichlet Accepts Zero Alpha Values and Violates Simplex Constraint

**Target**: `numpy.random.Generator.dirichlet`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `dirichlet` method accepts zero alpha parameters without validation and produces output that violates the fundamental mathematical constraint of the Dirichlet distribution - when all alpha values are zero, it returns an array that sums to 0 instead of 1.

## Property-Based Test

```python
import numpy as np
import numpy.random as npr
from hypothesis import given, strategies as st


@given(st.integers(min_value=2, max_value=10))
def test_dirichlet_all_zeros_violates_simplex_constraint(size):
    rng = npr.default_rng(42)

    alpha = np.zeros(size)
    result = rng.dirichlet(alpha)

    assert np.isclose(result.sum(), 1.0)

if __name__ == "__main__":
    test_dirichlet_all_zeros_violates_simplex_constraint()
```

<details>

<summary>
**Failing input**: `size=2`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 16, in <module>
    test_dirichlet_all_zeros_violates_simplex_constraint()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 7, in test_dirichlet_all_zeros_violates_simplex_constraint
    def test_dirichlet_all_zeros_violates_simplex_constraint(size):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 13, in test_dirichlet_all_zeros_violates_simplex_constraint
    assert np.isclose(result.sum(), 1.0)
           ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_dirichlet_all_zeros_violates_simplex_constraint(
    size=2,
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
import numpy.random as npr

rng = npr.default_rng(42)

# Test with all zeros
alpha_zeros = [0.0, 0.0, 0.0]
result = rng.dirichlet(alpha_zeros)

print(f"Alpha: {alpha_zeros}")
print(f"Result: {result}")
print(f"Sum: {result.sum()}")
print(f"Expected sum: 1.0")
print(f"Violation: Sum is {result.sum()} instead of 1.0")
```

<details>

<summary>
Output shows sum = 0.0 instead of required 1.0
</summary>
```
Alpha: [0.0, 0.0, 0.0]
Result: [0. 0. 0.]
Sum: 0.0
Expected sum: 1.0
Violation: Sum is 0.0 instead of 1.0
```
</details>

## Why This Is A Bug

The Dirichlet distribution is mathematically defined only for strictly positive alpha parameters (α_i > 0). The NumPy documentation describes the alpha parameter as "positive floats", where zero is not positive. The distribution's fundamental property is that samples must lie on the (n-1)-simplex, meaning they must sum to exactly 1.0.

When given all-zero alpha values, the function:
1. Accepts mathematically invalid parameters without error (violating the α_i > 0 requirement)
2. Returns output that sums to 0.0 instead of 1.0 (violating the simplex constraint)
3. Behaves inconsistently with NumPy's own `beta` distribution, which correctly raises `ValueError: a <= 0` for zero parameters

The current implementation validates only for negative values (`alpha < 0`) but not for zeros, despite the documentation stating parameters should be "positive floats". This creates a silent failure that could corrupt statistical analyses or simulations without warning.

## Relevant Context

The Dirichlet distribution is implemented by generating gamma variates and normalizing them:
- For each α_i, sample Y_i ~ Gamma(α_i, 1)
- Return X_i = Y_i / sum(Y)

When α_i = 0, the gamma distribution is undefined/degenerate and typically returns 0. If all alpha values are zero, all gamma samples are 0, leading to a 0/0 normalization that NumPy handles by returning all zeros.

The beta distribution (the 2-dimensional case of Dirichlet) already validates correctly:
- `rng.beta(0.0, 1.0)` raises `ValueError: a <= 0`
- This shows NumPy's intent to reject non-positive parameters

NumPy documentation: https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.dirichlet.html

## Proposed Fix

```diff
--- a/numpy/random/_generator.pyx
+++ b/numpy/random/_generator.pyx
@@ -4391,7 +4391,7 @@ cdef class Generator:
         d_arr = <np.ndarray>np.PyArray_FROM_OTF(alpha, np.NPY_DOUBLE, np.NPY_ALIGNED)

-        if np.any(np.less(d_arr, 0)):
-            raise ValueError('alpha < 0')
+        if np.any(np.less_equal(d_arr, 0)):
+            raise ValueError('alpha <= 0')

         shape = d_arr.shape
```