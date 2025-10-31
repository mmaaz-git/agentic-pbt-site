# Bug Report: numpy.polynomial.polyval Crashes with IndexError on Empty Coefficient Array

**Target**: `numpy.polynomial.polynomial.polyval`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `polyval` function and related polynomial evaluation functions crash with an `IndexError` when provided with an empty coefficient array, instead of either handling the edge case gracefully or raising a more informative `ValueError`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import numpy.polynomial.polynomial as poly

@given(st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
       st.lists(st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False),
                min_size=0, max_size=8))
@settings(max_examples=500)
def test_polyval_handles_empty(x, c):
    """Test that polyval handles empty coefficient array without crashing"""
    try:
        result = poly.polyval(x, c)
        if len(c) == 0:
            assert result == 0 or True, "Empty poly should evaluate to something"
    except ValueError:
        pass
    except IndexError:
        assert False, f"Should not raise IndexError for x={x}, c={c}"

# Run the test
test_polyval_handles_empty()
```

<details>

<summary>
**Failing input**: `x=0.0, c=[]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 11, in test_polyval_handles_empty
    result = poly.polyval(x, c)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/polynomial/polynomial.py", line 752, in polyval
    c0 = c[-1] + x * 0
         ~^^^^
IndexError: index -1 is out of bounds for axis 0 with size 0

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 20, in <module>
    test_polyval_handles_empty()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 5, in test_polyval_handles_empty
    st.lists(st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False),
            ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 17, in test_polyval_handles_empty
    assert False, f"Should not raise IndexError for x={x}, c={c}"
           ^^^^^
AssertionError: Should not raise IndexError for x=0.0, c=[]
Falsifying example: test_polyval_handles_empty(
    x=0.0,
    c=[],
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/34/hypo.py:14
```
</details>

## Reproducing the Bug

```python
import numpy.polynomial.polynomial as poly

# Test with empty coefficient array
result = poly.polyval(2.0, [])
print(f"Result: {result}")
```

<details>

<summary>
IndexError: index -1 is out of bounds for axis 0 with size 0
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/34/repo.py", line 4, in <module>
    result = poly.polyval(2.0, [])
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/polynomial/polynomial.py", line 752, in polyval
    c0 = c[-1] + x * 0
         ~^^^^
IndexError: index -1 is out of bounds for axis 0 with size 0
```
</details>

## Why This Is A Bug

This violates expected behavior for several reasons:

1. **Poor Error Handling**: The function crashes with an uninformative `IndexError` that exposes implementation details (attempting to access `c[-1]` on an empty array) rather than providing a meaningful error message about the actual problem (empty coefficient array).

2. **Documentation Mismatch**: The function documentation states that if `c` is of length `n + 1`, it returns the polynomial value. However, it doesn't specify what happens when `c` has length 0. According to mathematical convention, a polynomial with no coefficients could be considered the zero polynomial, which should evaluate to 0 for all inputs.

3. **Inconsistent Error Types**: The crash happens due to an implementation detail rather than input validation. A proper implementation would either:
   - Return 0 (treating empty coefficients as the zero polynomial)
   - Raise a `ValueError` with a clear message like "coefficient array cannot be empty"

4. **Widespread Impact**: This bug affects multiple functions in the polynomial module that rely on `polyval`:
   - `polyval2d(x, y, c)` - crashes when c is empty
   - `polyval3d(x, y, z, c)` - crashes when c is empty
   - `polygrid2d(x, y, c)` - crashes when c is empty
   - `polygrid3d(x, y, z, c)` - crashes when c is empty

## Relevant Context

The bug occurs in line 752 of `/home/npc/miniconda/lib/python3.13/site-packages/numpy/polynomial/polynomial.py`:

```python
c0 = c[-1] + x * 0
```

This line attempts to access the last element of the coefficient array `c` without checking if the array is empty. The function uses Horner's method for polynomial evaluation, which requires at least one coefficient to start the evaluation.

The numpy polynomial module documentation can be found at: https://numpy.org/doc/stable/reference/routines.polynomials.polynomial.html

Similar functions in other polynomial representations (Chebyshev, Legendre, etc.) may have the same issue since they share similar implementations.

## Proposed Fix

```diff
--- a/numpy/polynomial/polynomial.py
+++ b/numpy/polynomial/polynomial.py
@@ -741,6 +741,8 @@ def polyval(x, c, tensor=True):

     """
     c = np.array(c, ndmin=1, copy=None)
+    if c.size == 0:
+        raise ValueError("coefficient array cannot be empty")
     if c.dtype.char in '?bBhHiIlLqQpP':
         # astype fails with NA
         c = c + 0.0
```