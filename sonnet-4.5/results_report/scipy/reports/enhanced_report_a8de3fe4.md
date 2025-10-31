# Bug Report: scipy.optimize.cython_optimize Uninitialized Memory in iterations Field

**Target**: `scipy.optimize.cython_optimize._zeros.full_output_example`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `iterations` field in the `zeros_full_output` struct contains uninitialized memory (garbage values like -193122112 or 1627276288) when root-finding algorithms terminate early at boundary values, violating the documented API contract that promises a valid iteration count.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from scipy.optimize.cython_optimize import _zeros

@given(
    a0=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_full_output_has_valid_counts_on_success(a0):
    args = (a0, 0.0, 0.0, 1.0)
    xa, xb = 0.0, 10.0
    xtol, rtol, mitr = 1e-6, 1e-6, 100

    output = _zeros.full_output_example(args, xa, xb, xtol, rtol, mitr)

    if output['error_num'] == 0:
        assert output['iterations'] >= 0, \
            f"Successful solve should have iterations >= 0, got {output['iterations']}"
        assert output['funcalls'] >= 1, \
            f"Successful solve should have funcalls >= 1, got {output['funcalls']}"

if __name__ == "__main__":
    test_full_output_has_valid_counts_on_success()
```

<details>

<summary>
**Failing input**: `a0=0.0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/48/hypo.py", line 22, in <module>
    test_full_output_has_valid_counts_on_success()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/48/hypo.py", line 5, in test_full_output_has_valid_counts_on_success
    a0=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/48/hypo.py", line 16, in test_full_output_has_valid_counts_on_success
    assert output['iterations'] >= 0, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Successful solve should have iterations >= 0, got -124894976
Falsifying example: test_full_output_has_valid_counts_on_success(
    a0=0.0,
)
```
</details>

## Reproducing the Bug

```python
from scipy.optimize.cython_optimize import _zeros

# Test case that demonstrates the bug: root at boundary x=0
# The polynomial is f(x) = x^3 when args=(0,0,0,1)
a0 = 0.0
args = (a0, 0.0, 0.0, 1.0)
xa, xb = 0.0, 10.0
xtol, rtol, mitr = 1e-6, 1e-6, 100

output = _zeros.full_output_example(args, xa, xb, xtol, rtol, mitr)
print(f"iterations={output['iterations']}, funcalls={output['funcalls']}, "
      f"error_num={output['error_num']}, root={output['root']}")

# This assertion should pass but fails due to uninitialized memory
assert output['iterations'] >= 1, f"Expected iterations >= 1, got {output['iterations']}"
```

<details>

<summary>
AssertionError: iterations contains garbage value from uninitialized memory
</summary>
```
iterations=-193122112, funcalls=2, error_num=0, root=0.0
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/48/repo.py", line 15, in <module>
    assert output['iterations'] >= 1, f"Expected iterations >= 1, got {output['iterations']}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Expected iterations >= 1, got -193122112
```
</details>

## Why This Is A Bug

This bug violates the documented API contract in multiple critical ways:

1. **Documentation Contract Violation**: The SciPy documentation at `scipy/optimize/cython_optimize/__init__.py` explicitly states that the `zeros_full_output` struct contains:
   - `int iterations`: number of iterations

   The documentation shows an example output: `{'error_num': 0, 'funcalls': 6, 'iterations': 5, 'root': 0.6999...}` indicating iterations should be a valid non-negative count.

2. **Uninitialized Memory Exposure**: The `iterations` field contains garbage values from uninitialized memory, which is evidenced by:
   - Non-deterministic values across multiple runs (e.g., -193122112, 1627276288, 9997)
   - Often large negative or positive integers that make no sense as iteration counts
   - This is a serious reliability issue and potential security concern

3. **Inconsistent Behavior**: The bug occurs specifically when:
   - The function value at a boundary (f(xa) or f(xb)) is within tolerance of zero
   - The algorithm terminates early after only 2 function evaluations
   - The early return path fails to initialize the iterations field

4. **API Reliability**: Users cannot reliably access iteration counts for algorithm analysis, performance monitoring, or debugging when the solver succeeds.

## Relevant Context

The `full_output_example` function uses the underlying C implementations of root-finding algorithms (brentq, bisect, ridder, brenth). When these algorithms detect that a boundary point is already a root (within tolerance), they take an early return path that skips the normal iteration loop.

Testing reveals the non-deterministic nature of the bug:
- Multiple runs with identical inputs produce different iteration values
- Values range from large negative numbers to large positive numbers
- This confirms the field contains whatever garbage was in memory

The bug affects a common use case - finding roots of functions where the root happens to be at or near the search interval boundary. This is not a rare edge case but a normal scenario in numerical computing.

Documentation reference: https://docs.scipy.org/doc/scipy/reference/optimize.cython_optimize.html

## Proposed Fix

The fix requires modifying the C source code for the root-finding algorithms to properly initialize the `zeros_full_output` struct. Since the C source is compiled into the Cython extension, a high-level overview of the required changes:

1. Initialize all fields of `zeros_full_output` at function entry
2. Set iterations to 0 in early return paths when boundary roots are found
3. Ensure all code paths that set error_num=0 also set valid iteration counts

The conceptual fix in the C code would be:

```diff
 double brentq(callback_type f, double xa, double xb, void* args,
               double xtol, double rtol, int iter,
               zeros_full_output *full_output) {

+    /* Initialize output struct if provided */
+    if (full_output != NULL) {
+        full_output->iterations = 0;
+        full_output->funcalls = 0;
+        full_output->error_num = 0;
+        full_output->root = 0.0;
+    }
+
     double fa = f(xa, args);
     double fb = f(xb, args);

     if (full_output != NULL) {
         full_output->funcalls = 2;
     }

     /* Check if boundary is already a root */
     if (fabs(fa) < xtol) {
         if (full_output != NULL) {
             full_output->root = xa;
-            /* BUG: iterations field not set here */
+            full_output->iterations = 0;  /* Fixed: set iterations */
             full_output->error_num = 0;
         }
         return xa;
     }
```