# Bug Report: pandas.arrays.IntegerArray.__pow__ Fails for Base=1 with Negative Exponent

**Target**: `pandas.arrays.IntegerArray.__pow__`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

IntegerArray raises ValueError when computing 1^(-n) for negative integer n, even though this operation is mathematically well-defined (always equals 1) and pandas has unreachable special-case code to handle it.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
import pandas as pd


@given(st.lists(st.integers(min_value=-10, max_value=10) | st.none(), min_size=1, max_size=30))
@settings(max_examples=500)
def test_integerarray_one_pow_x_is_one(exponents):
    arr = pd.array(exponents, dtype="Int64")
    base = pd.array([1] * len(arr), dtype="Int64")
    result = base ** arr

    for i in range(len(result)):
        if pd.notna(result[i]):
            assert result[i] == 1


if __name__ == "__main__":
    test_integerarray_one_pow_x_is_one()
```

<details>

<summary>
**Failing input**: `exponents=[-1]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/36/hypo.py", line 18, in <module>
    test_integerarray_one_pow_x_is_one()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/36/hypo.py", line 6, in test_integerarray_one_pow_x_is_one
    @settings(max_examples=500)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/36/hypo.py", line 10, in test_integerarray_one_pow_x_is_one
    result = base ** arr
             ~~~~~^^~~~~
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/ops/common.py", line 76, in new_method
    return method(self, other)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arraylike.py", line 242, in __pow__
    return self._arith_method(other, operator.pow)
           ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/masked.py", line 807, in _arith_method
    result = pd_op(self._data, other)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/ops/array_ops.py", line 283, in arithmetic_op
    res_values = _na_arithmetic_op(left, right, op)  # type: ignore[arg-type]
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/ops/array_ops.py", line 218, in _na_arithmetic_op
    result = func(left, right)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/computation/expressions.py", line 242, in evaluate
    return _evaluate(op, op_str, a, b)  # type: ignore[misc]
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/computation/expressions.py", line 73, in _evaluate_standard
    return op(a, b)
ValueError: Integers to negative integer powers are not allowed.
Falsifying example: test_integerarray_one_pow_x_is_one(
    exponents=[-1],
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/pandas/core/ops/array_ops.py:219
```
</details>

## Reproducing the Bug

```python
import pandas as pd

# Reproducing the bug: IntegerArray with base=1 and negative exponent
base = pd.array([1], dtype="Int64")
exponent = pd.array([-1], dtype="Int64")

print("Computing: base ** exponent")
print(f"base = {base}")
print(f"exponent = {exponent}")

try:
    result = base ** exponent
    print(f"result = {result}")
except ValueError as e:
    print(f"ValueError raised: {e}")
```

<details>

<summary>
ValueError raised when computing 1 ** (-1)
</summary>
```
Computing: base ** exponent
base = <IntegerArray>
[1]
Length: 1, dtype: Int64
exponent = <IntegerArray>
[-1]
Length: 1, dtype: Int64
ValueError raised: Integers to negative integer powers are not allowed.
```
</details>

## Why This Is A Bug

This violates expected mathematical behavior where 1^n = 1 for any integer n (including negative values). The operation 1^(-1) = 1/1 = 1 is mathematically well-defined and has an exact integer result.

More importantly, pandas already contains special-case code in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/arrays/masked.py:811` that explicitly handles this case with the comment "# 1 ** x is 1." However, this optimization code is unreachable because it's placed AFTER the power operation (line 807) that raises the ValueError. The bug is an implementation error where the special case check happens too late in the execution flow.

## Relevant Context

The bug occurs in the `_arith_method` function of the `BaseMaskedArray` class. The relevant code section shows:
- Line 807: `result = pd_op(self._data, other)` - This raises the ValueError for integer arrays with negative exponents
- Line 811: `mask = np.where((self._data == 1) & ~self._mask, False, mask)` - Special case for "1 ** x is 1" that never gets reached

This behavior follows NumPy's design decision (numpy.power documentation states: "An integer type raised to a negative integer power will raise a ValueError"), but pandas attempted to override this for the mathematically correct case of 1^x = 1. The same issue affects the `rpow` operation (lines 819-823).

Documentation: Neither pandas.arrays.IntegerArray nor pandas.Series.pow documentation explicitly addresses this edge case behavior.

## Proposed Fix

Move the special case handling for base=1 and exponent=0 before the actual computation to avoid the ValueError:

```diff
--- a/pandas/core/arrays/masked.py
+++ b/pandas/core/arrays/masked.py
@@ -799,13 +799,31 @@ class BaseMaskedArray(OpsMixin, ExtensionArray):
             # Make sure we do this before the "pow" mask checks
             #  to get an expected exception message on shape mismatch.
             if self.dtype.kind in "iu" and op_name in ["floordiv", "mod"]:
                 # TODO(GH#30188) ATM we don't match the behavior of non-masked
                 #  types with respect to floordiv-by-zero
                 pd_op = op

+            # Handle special cases for integer power operations to avoid ValueError
+            if op_name in ["pow", "rpow"] and self.dtype.kind in "iu":
+                # For integer dtypes, 1**x=1 and x**0=1, but numpy raises error
+                # for negative exponents. Compute these cases separately.
+                result = np.empty_like(self._data)
+                if op_name == "pow":
+                    safe_mask = (self._data != 1) & (other != 0)
+                    result[~safe_mask] = 1
+                    if safe_mask.any():
+                        result[safe_mask] = pd_op(self._data[safe_mask], other[safe_mask] if not is_scalar(other) else other)
+                else:  # rpow
+                    safe_mask = (other != 1) & (self._data != 0)
+                    result[~safe_mask] = 1
+                    if safe_mask.any():
+                        result[safe_mask] = pd_op(self._data[safe_mask], other)
+            else:
             with np.errstate(all="ignore"):
                 result = pd_op(self._data, other)

         if op_name == "pow":
```