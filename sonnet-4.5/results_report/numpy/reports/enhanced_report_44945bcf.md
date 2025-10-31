# Bug Report: numpy.array_equiv Violates Reflexivity with NaN Values

**Target**: `numpy.array_equiv`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The function `numpy.array_equiv(arr, arr)` incorrectly returns `False` when `arr` contains NaN values, violating the mathematical property of reflexivity that requires any element to be equivalent to itself.

## Property-Based Test

```python
from hypothesis import given, settings, Verbosity
import hypothesis.strategies as st
import hypothesis.extra.numpy as npst
import numpy as np

@given(npst.arrays(dtype=npst.floating_dtypes(), shape=npst.array_shapes()))
@settings(max_examples=100, verbosity=Verbosity.verbose)
def test_numpy_array_equiv_reflexivity(arr):
    """Test that array_equiv satisfies reflexivity: array_equiv(x, x) should always be True"""
    # An array should always be equivalent to itself
    result = np.array_equiv(arr, arr)
    assert result, f"array_equiv reflexivity violated: array_equiv({arr!r}, {arr!r}) returned False"

if __name__ == "__main__":
    # Run the test and let it find the failing case
    test_numpy_array_equiv_reflexivity()
```

<details>

<summary>
**Failing input**: `array([nan], dtype=float16)`
</summary>
```
Trying example: test_numpy_array_equiv_reflexivity(
    arr=array([nan], dtype=float16),
)
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 12, in test_numpy_array_equiv_reflexivity
    assert result, f"array_equiv reflexivity violated: array_equiv({arr!r}, {arr!r}) returned False"
           ^^^^^^
AssertionError: array_equiv reflexivity violated: array_equiv(array([nan], dtype=float16), array([nan], dtype=float16)) returned False

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 16, in <module>
    test_numpy_array_equiv_reflexivity()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 7, in test_numpy_array_equiv_reflexivity
    @settings(max_examples=100, verbosity=Verbosity.verbose)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 12, in test_numpy_array_equiv_reflexivity
    assert result, f"array_equiv reflexivity violated: array_equiv({arr!r}, {arr!r}) returned False"
           ^^^^^^
AssertionError: array_equiv reflexivity violated: array_equiv(array([nan], dtype=float16), array([nan], dtype=float16)) returned False
Falsifying example: test_numpy_array_equiv_reflexivity(
    arr=array([nan], dtype=float16),
)
```
</details>

## Reproducing the Bug

```python
import numpy as np

# Test case 1: Array with NaN should be equivalent to itself (reflexivity)
arr = np.array([np.nan])
result = np.array_equiv(arr, arr)
print(f"Test 1 - Self-equivalence with NaN:")
print(f"  arr = {arr}")
print(f"  np.array_equiv(arr, arr) = {result}")
print(f"  Expected: True (reflexivity property)")
print()

# Test case 2: Array without NaN works correctly
arr2 = np.array([1.0, 2.0])
result2 = np.array_equiv(arr2, arr2)
print(f"Test 2 - Self-equivalence without NaN:")
print(f"  arr2 = {arr2}")
print(f"  np.array_equiv(arr2, arr2) = {result2}")
print(f"  Expected: True (works correctly)")
print()

# Test case 3: Multiple NaN values
arr3 = np.array([np.nan, np.nan, 1.0])
result3 = np.array_equiv(arr3, arr3)
print(f"Test 3 - Self-equivalence with multiple NaN:")
print(f"  arr3 = {arr3}")
print(f"  np.array_equiv(arr3, arr3) = {result3}")
print(f"  Expected: True (reflexivity property)")
```

<details>

<summary>
Output showing reflexivity violation with NaN
</summary>
```
Test 1 - Self-equivalence with NaN:
  arr = [nan]
  np.array_equiv(arr, arr) = False
  Expected: True (reflexivity property)

Test 2 - Self-equivalence without NaN:
  arr2 = [1. 2.]
  np.array_equiv(arr2, arr2) = True
  Expected: True (works correctly)

Test 3 - Self-equivalence with multiple NaN:
  arr3 = [nan nan  1.]
  np.array_equiv(arr3, arr3) = False
  Expected: True (reflexivity property)
```
</details>

## Why This Is A Bug

The function `numpy.array_equiv` violates the fundamental mathematical property of reflexivity, which requires that any element must be equivalent to itself. This violation occurs specifically when arrays contain NaN values.

**Mathematical Context:**
- The term "equivalence" in mathematics implies an equivalence relation, which must satisfy three properties:
  1. **Reflexivity**: `a ~ a` (every element is equivalent to itself) - **VIOLATED**
  2. **Symmetry**: if `a ~ b` then `b ~ a`
  3. **Transitivity**: if `a ~ b` and `b ~ c` then `a ~ c`

**Current Implementation Issue:**
- The function uses simple element-wise equality comparison: `(a1 == a2).all()`
- Following IEEE 754 standard, `NaN != NaN` returns `False`
- Therefore, when comparing an array containing NaN to itself, the function incorrectly returns `False`

**Impact on Users:**
- Violates the principle of least surprise - users reasonably expect `array_equiv(x, x)` to always return `True`
- NaN values are common in scientific computing (missing data, invalid operations, etc.)
- The function name strongly implies it tests for equivalence, setting user expectations

**Inconsistency with NumPy's Design:**
- NumPy's `array_equal` function already addresses this issue with an `equal_nan` parameter
- `array_equal(arr, arr, equal_nan=True)` correctly returns `True` for arrays with NaN
- The lack of similar functionality in `array_equiv` creates an inconsistency in NumPy's API

## Relevant Context

**Documentation Analysis:**
The current documentation for `numpy.array_equiv` states:
- "Returns True if input arrays are shape consistent and all elements equal"
- No mention of NaN handling or the reflexivity violation
- No `equal_nan` parameter like `array_equal` has

**Related Functions:**
- `numpy.array_equal`: Has an `equal_nan` parameter (default=False) to handle NaN comparisons
- Documentation link: https://numpy.org/doc/stable/reference/generated/numpy.array_equiv.html
- Source code location: `/numpy/_core/numeric.py:2615`

**Workarounds Available:**
1. Use `numpy.array_equal(a1, a2, equal_nan=True)` when shape broadcasting is not needed
2. Pre-check for NaN values and handle them separately
3. Use object identity check: `a1 is a2` for self-comparison cases

## Proposed Fix

Add an `equal_nan` parameter to `array_equiv` similar to `array_equal`, defaulting to `True` to preserve reflexivity:

```diff
--- a/numpy/_core/numeric.py
+++ b/numpy/_core/numeric.py
@@ -2610,11 +2610,11 @@ def isclose(a, b, rtol=1.e-5, atol=1.e-8, equal_nan=False):
     return within_tol


-def _array_equiv_dispatcher(a1, a2):
+def _array_equiv_dispatcher(a1, a2, equal_nan=None):
     return (a1, a2)


 @array_function_dispatch(_array_equiv_dispatcher)
-def array_equiv(a1, a2):
+def array_equiv(a1, a2, equal_nan=True):
     """
     Returns True if input arrays are shape consistent and all elements equal.
@@ -2625,6 +2625,10 @@ def array_equiv(a1, a2):
     ----------
     a1, a2 : array_like
         Input arrays.
+    equal_nan : bool, optional
+        Whether to compare NaN's as equal. If True, NaN's in `a1` will be
+        considered equal to NaN's in `a2`. Default is True to preserve
+        reflexivity (an array is always equivalent to itself).

     Returns
     -------
@@ -2658,7 +2662,16 @@ def array_equiv(a1, a2):
     except Exception:
         return False

-    return builtins.bool(asanyarray(a1 == a2).all())
+    # Fast path for non-NaN case or when equal_nan is False
+    if not equal_nan:
+        return builtins.bool(asanyarray(a1 == a2).all())
+
+    # Check if arrays are equal, handling NaN values
+    eq = (a1 == a2)
+    if eq.all():
+        return True
+
+    # Check if differences are due to NaN values at the same positions
+    return builtins.bool(eq | (isnan(a1) & isnan(a2))).all()


 def _astype_dispatcher(x, dtype, /, *, copy=None, device=None):
```

**Note:** Setting `equal_nan=True` as default preserves reflexivity while changing existing behavior. An alternative approach would be to default to `False` for backward compatibility but explicitly check for object identity (`a1 is a2`) to preserve reflexivity in self-comparison cases.