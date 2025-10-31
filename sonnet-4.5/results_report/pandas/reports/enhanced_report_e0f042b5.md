# Bug Report: pandas.core.ops.array_ops._masked_arith_op rpow with base=1 returns NaN instead of 1

**Target**: `pandas.core.ops.array_ops._masked_arith_op`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The function `_masked_arith_op` incorrectly returns NaN when computing `1 ** x` (rpow with base=1), violating the fundamental mathematical property that 1 raised to any power equals 1. The implementation contradicts its own source code comment and breaks IEEE 754 floating-point standards.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
from pandas.core import roperator
from pandas.core.ops.array_ops import _masked_arith_op


@settings(max_examples=500)
@given(
    exponent=st.floats(allow_nan=False, allow_infinity=False, min_value=-10, max_value=10)
)
def test_masked_arith_op_rpow_base_one_should_return_one(exponent):
    x = np.array([exponent, np.nan, exponent], dtype=object)
    y = 1.0

    result = _masked_arith_op(x, y, roperator.rpow)

    assert result[0] == 1.0, f"Expected 1.0 for 1.0**{exponent}, got {result[0]}"
    assert result[2] == 1.0, f"Expected 1.0 for 1.0**{exponent}, got {result[2]}"

if __name__ == "__main__":
    test_masked_arith_op_rpow_base_one_should_return_one()
```

<details>

<summary>
**Failing input**: `exponent=0.0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 21, in <module>
    test_masked_arith_op_rpow_base_one_should_return_one()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 8, in test_masked_arith_op_rpow_base_one_should_return_one
    @given(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 17, in test_masked_arith_op_rpow_base_one_should_return_one
    assert result[0] == 1.0, f"Expected 1.0 for 1.0**{exponent}, got {result[0]}"
           ^^^^^^^^^^^^^^^^
AssertionError: Expected 1.0 for 1.0**0.0, got nan
Falsifying example: test_masked_arith_op_rpow_base_one_should_return_one(
    exponent=0.0,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from pandas.core import roperator
from pandas.core.ops.array_ops import _masked_arith_op

# Test case 1: Basic rpow with base 1
x = np.array([0.0, 1.0, 2.0], dtype=object)
y = 1.0

result = _masked_arith_op(x, y, roperator.rpow)

print(f"Test 1: Basic rpow with base 1")
print(f"x = {x}")
print(f"y = {y}")
print(f"Result: {result}")
print(f"Expected: [1.0, 1.0, 1.0]")
print()

# Test case 2: rpow with base 1 and NaN values
x_with_nan = np.array([0.0, np.nan, 2.0], dtype=object)
y = 1.0

result_with_nan = _masked_arith_op(x_with_nan, y, roperator.rpow)

print(f"Test 2: rpow with base 1 and NaN values")
print(f"x = {x_with_nan}")
print(f"y = {y}")
print(f"Result: {result_with_nan}")
print(f"Expected: [1.0, nan, 1.0]")
print()

# Verify NumPy's behavior for comparison
print("NumPy's behavior for comparison:")
print(f"1.0 ** 0.0 = {1.0 ** 0.0}")
print(f"1.0 ** 1.0 = {1.0 ** 1.0}")
print(f"1.0 ** 2.0 = {1.0 ** 2.0}")
print(f"1.0 ** np.nan = {1.0 ** np.nan}")
```

<details>

<summary>
Output shows NaN instead of 1.0 for all rpow operations with base 1
</summary>
```
Test 1: Basic rpow with base 1
x = [0.0 1.0 2.0]
y = 1.0
Result: [nan nan nan]
Expected: [1.0, 1.0, 1.0]

Test 2: rpow with base 1 and NaN values
x = [0.0 nan 2.0]
y = 1.0
Result: [nan nan nan]
Expected: [1.0, nan, 1.0]

NumPy's behavior for comparison:
1.0 ** 0.0 = 1.0
1.0 ** 1.0 = 1.0
1.0 ** 2.0 = 1.0
1.0 ** np.nan = 1.0
```
</details>

## Why This Is A Bug

This violates expected mathematical behavior and contradicts the implementation's own documentation. The function contains a comment at line 175 stating "1 ** np.nan is 1. So we have to unmask those", explicitly acknowledging that 1 raised to any power (including NaN) should equal 1. However, the implementation at line 179 incorrectly sets the mask to False where y==1, which prevents the operation from being computed and causes NaN to be filled instead.

The bug violates:
1. **Mathematical principles**: 1^x = 1 is a fundamental property for all real x
2. **IEEE 754-2008/2019 standard**: Specifies that pow(+1, y) = 1 for any y, "even a quiet NaN"
3. **NumPy consistency**: NumPy correctly returns 1.0 for `1.0 ** np.nan`, but pandas returns NaN
4. **Developer intent**: The source code comment clearly states the correct behavior

## Relevant Context

The bug is located in `/pandas/core/ops/array_ops.py` at lines 178-179. The function has special handling for power operations to ensure mathematical correctness, but the rpow case is implemented backwards.

Current problematic code:
```python
elif op is roperator.rpow:
    mask = np.where(y == 1, False, mask)
```

This sets mask to False where y==1, preventing computation. Line 184 then fills these positions with NaN:
```python
np.putmask(result, ~mask, np.nan)
```

The roperator.rpow function is defined in `/pandas/core/roperator.py` as:
```python
def rpow(left, right):
    return right**left
```

So when called as `_masked_arith_op(x, 1.0, roperator.rpow)`, it should compute `1.0 ** x[i]` for each element.

## Proposed Fix

```diff
--- a/pandas/core/ops/array_ops.py
+++ b/pandas/core/ops/array_ops.py
@@ -176,7 +176,7 @@ def _masked_arith_op(x: np.ndarray, y, op):
         if op is pow:
             mask = np.where(x == 1, False, mask)
         elif op is roperator.rpow:
-            mask = np.where(y == 1, False, mask)
+            # For rpow with base 1, keep mask True to allow computation
+            # NumPy correctly handles 1**x = 1 for all x including NaN
+            pass

         if mask.any():
             result[mask] = op(xrav[mask], y)
```