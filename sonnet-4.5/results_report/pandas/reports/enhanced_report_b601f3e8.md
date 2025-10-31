# Bug Report: pandas.core.ops.kleene_xor Commutativity Violation with NA Values

**Target**: `pandas.core.ops.kleene_xor`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `kleene_xor` function violates commutativity by returning different underlying result values for `True ^ NA` vs `NA ^ True` when both are masked as NA, contradicting the explicit code comment that states "A ^ B == B ^ A".

## Property-Based Test

```python
import numpy as np
from pandas._libs import missing as libmissing
import pandas.core.ops as ops
from hypothesis import given, strategies as st, settings


@given(left_val=st.booleans())
@settings(max_examples=10)
def test_kleene_xor_na_commutativity_full(left_val):
    left_with_value = np.array([left_val])
    mask_with_value = np.array([False])

    left_with_na = np.array([False])
    mask_with_na = np.array([True])

    result_value_na, mask_value_na = ops.kleene_xor(left_with_value, libmissing.NA, mask_with_value, None)
    result_na_value, mask_na_value = ops.kleene_xor(left_with_na, left_val, mask_with_na, None)

    assert mask_value_na[0] == mask_na_value[0]
    assert result_value_na[0] == result_na_value[0]


if __name__ == "__main__":
    test_kleene_xor_na_commutativity_full()
```

<details>

<summary>
**Failing input**: `left_val=True`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 24, in <module>
    test_kleene_xor_na_commutativity_full()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 8, in test_kleene_xor_na_commutativity_full
    @settings(max_examples=10)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 20, in test_kleene_xor_na_commutativity_full
    assert result_value_na[0] == result_na_value[0]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_kleene_xor_na_commutativity_full(
    left_val=True,
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from pandas._libs import missing as libmissing
import pandas.core.ops as ops

# Test case 1: True ^ NA
left_true = np.array([True])
mask_false = np.array([False])

# Test case 2: NA ^ True
left_na = np.array([False])
mask_true = np.array([True])

# Execute operations
result1, mask1 = ops.kleene_xor(left_true, libmissing.NA, mask_false, None)
result2, mask2 = ops.kleene_xor(left_na, True, mask_true, None)

print(f'True ^ NA: result={result1[0]}, mask={mask1[0]}')
print(f'NA ^ True: result={result2[0]}, mask={mask2[0]}')

# Check if results are the same (they should be for commutative operations)
if result1[0] == result2[0]:
    print("✓ Results are equal (commutative)")
else:
    print("✗ Results differ - commutativity violated!")
    print(f"  True ^ NA returns result={result1[0]}")
    print(f"  NA ^ True returns result={result2[0]}")

# This assertion will fail, demonstrating the bug
assert result1[0] == result2[0], f"Commutativity violated: {result1[0]} != {result2[0]}"
```

<details>

<summary>
AssertionError: Commutativity violated - XOR operation is not commutative with NA values
</summary>
```
True ^ NA: result=False, mask=True
NA ^ True: result=True, mask=True
✗ Results differ - commutativity violated!
  True ^ NA returns result=False
  NA ^ True returns result=True
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/37/repo.py", line 29, in <module>
    assert result1[0] == result2[0], f"Commutativity violated: {result1[0]} != {result2[0]}"
           ^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Commutativity violated: False != True
```
</details>

## Why This Is A Bug

The `kleene_xor` function explicitly documents that XOR is commutative in two places:

1. **Line 105** in `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/ops/mask_ops.py` contains the comment: `# A ^ B == B ^ A`

2. **Lines 103-107** explain that arguments can be safely swapped because of commutativity: "To reduce the number of cases, we ensure that `left` & `left_mask` always come from an array, not a scalar. This is safe, since A ^ B == B ^ A"

However, the implementation violates this documented property. When computing `True ^ NA` vs `NA ^ True`:

- `True ^ NA`: Line 114 sets `result = np.zeros_like(left)`, resulting in `False`
- `NA ^ True`: After argument swapping (line 107), the left array contains the NA placeholder value (False), so line 116 computes `False ^ True = True`

While both operations correctly indicate the result is NA via `mask=True`, the underlying result array values differ (False vs True). This violates the mathematical property of commutativity that the code explicitly claims to maintain. Although most code only checks masks and ignores underlying values when masked, this inconsistency could cause subtle bugs if any code inspects result values even when masked, or if the implementation changes in the future.

## Relevant Context

The bug occurs due to an asymmetry in how the function handles NA values:
- When `right` is NA (line 114), the result is forced to zeros regardless of left values
- When arguments are swapped and NA ends up on the left, the actual XOR computation happens (line 116) using the NA placeholder value

The same pattern exists in `kleene_and` (line 164) but not in `kleene_or` (line 51 uses `left.copy()` instead of zeros).

Documentation references:
- Function source: `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/ops/mask_ops.py:76-126`
- Export location: `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/core/ops/__init__.py:26`

## Proposed Fix

```diff
--- a/pandas/core/ops/mask_ops.py
+++ b/pandas/core/ops/mask_ops.py
@@ -111,7 +111,7 @@ def kleene_xor(

     raise_for_nan(right, method="xor")
     if right is libmissing.NA:
-        result = np.zeros_like(left)
+        result = left.copy()
     else:
         result = left ^ right
```