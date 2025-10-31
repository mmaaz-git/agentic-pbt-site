# Bug Report: pandas.core.ops Kleene Logic Functions Infinite Recursion

**Target**: `pandas.core.ops.kleene_and`, `pandas.core.ops.kleene_or`, `pandas.core.ops.kleene_xor`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The Kleene logic functions (`kleene_and`, `kleene_or`, `kleene_xor`) enter infinite recursion when both `left_mask` and `right_mask` parameters are `None`, causing a `RecursionError` crash instead of validating this documented precondition.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings, example
from pandas.core.ops import kleene_and

@given(st.lists(st.booleans(), min_size=1, max_size=50),
       st.lists(st.booleans(), min_size=1, max_size=50))
@example([True], [True])  # Add a specific example to ensure it fails
@settings(max_examples=10)  # Reduce examples to get cleaner output
def test_kleene_and_without_na(left_vals, right_vals):
    min_len = min(len(left_vals), len(right_vals))
    left = np.array(left_vals[:min_len])
    right = np.array(right_vals[:min_len])
    result, mask = kleene_and(left, right, None, None)
    expected = left & right
    assert np.array_equal(result, expected)

if __name__ == "__main__":
    # Run the test
    test_kleene_and_without_na()
```

<details>

<summary>
**Failing input**: `left_vals=[True]`, `right_vals=[True]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/2/hypo.py", line 19, in <module>
    test_kleene_and_without_na()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/2/hypo.py", line 6, in test_kleene_and_without_na
    st.lists(st.booleans(), min_size=1, max_size=50))
            ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/2/hypo.py", line 13, in test_kleene_and_without_na
    result, mask = kleene_and(left, right, None, None)
                   ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/ops/mask_ops.py", line 157, in kleene_and
    return kleene_and(right, left, right_mask, left_mask)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/ops/mask_ops.py", line 157, in kleene_and
    return kleene_and(right, left, right_mask, left_mask)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/ops/mask_ops.py", line 157, in kleene_and
    return kleene_and(right, left, right_mask, left_mask)
  [Previous line repeated 1997 more times]
RecursionError: maximum recursion depth exceeded
Falsifying explicit example: test_kleene_and_without_na(
    left_vals=[True],
    right_vals=[True],
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from pandas.core.ops import kleene_and, kleene_or, kleene_xor

# Create simple test arrays
left = np.array([True, False])
right = np.array([True, True])

# Test kleene_and with both masks as None
print("Testing kleene_and with both masks as None:")
try:
    result, mask = kleene_and(left, right, None, None)
    print(f"Result: {result}, Mask: {mask}")
except RecursionError as e:
    print(f"RecursionError: maximum recursion depth exceeded")
except Exception as e:
    print(f"Error: {e}")

# Test kleene_or with both masks as None
print("\nTesting kleene_or with both masks as None:")
try:
    result, mask = kleene_or(left, right, None, None)
    print(f"Result: {result}, Mask: {mask}")
except RecursionError as e:
    print(f"RecursionError: maximum recursion depth exceeded")
except Exception as e:
    print(f"Error: {e}")

# Test kleene_xor with both masks as None
print("\nTesting kleene_xor with both masks as None:")
try:
    result, mask = kleene_xor(left, right, None, None)
    print(f"Result: {result}, Mask: {mask}")
except RecursionError as e:
    print(f"RecursionError: maximum recursion depth exceeded")
except Exception as e:
    print(f"Error: {e}")
```

<details>

<summary>
RecursionError: All three functions crash with maximum recursion depth exceeded
</summary>
```
Testing kleene_and with both masks as None:
RecursionError: maximum recursion depth exceeded

Testing kleene_or with both masks as None:
RecursionError: maximum recursion depth exceeded

Testing kleene_xor with both masks as None:
RecursionError: maximum recursion depth exceeded
```
</details>

## Why This Is A Bug

This violates expected behavior in several critical ways:

1. **Documentation Contract Violation**: Each function's docstring explicitly states "Only one of these may be None" for the mask parameters, but the functions do not validate or enforce this precondition. Instead of raising a clear error, they crash with infinite recursion.

2. **Infinite Recursion Logic Flaw**: When both masks are `None`, the functions enter an infinite loop:
   - In `kleene_and`: Line 156 checks `if left_mask is None`, then line 157 calls `return kleene_and(right, left, right_mask, left_mask)`
   - Since `right_mask` is also `None`, this becomes `kleene_and(right, left, None, None)`
   - The swapped call again finds `left_mask is None` and swaps back to `kleene_and(left, right, None, None)`
   - This continues infinitely until Python's recursion limit is hit

3. **Unacceptable Failure Mode**: Even when users violate API preconditions, functions should fail gracefully with informative errors, not crash the entire program with a `RecursionError`. This makes debugging difficult and can cause unexpected production failures.

4. **All Three Functions Affected**: The same logic flaw exists in `kleene_and` (line 156-157), `kleene_or` (line 42-43), and `kleene_xor` (line 106-107), making this a systematic design issue rather than an isolated bug.

## Relevant Context

These functions are part of pandas' core operations for handling nullable boolean arrays using three-valued (Kleene) logic. They're typically used internally by pandas when performing boolean operations on nullable data types. The mask parameters indicate which values are NA/missing.

The functions attempt to optimize by ensuring the left operand is always an array (swapping arguments if needed), but fail to handle the edge case where neither operand has a mask, leading to infinite mutual recursion.

Relevant source code location: `/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/ops/mask_ops.py`

## Proposed Fix

```diff
--- a/pandas/core/ops/mask_ops.py
+++ b/pandas/core/ops/mask_ops.py
@@ -38,6 +38,10 @@ def kleene_or(
     """
+    # Validate precondition: at least one mask must be provided
+    if left_mask is None and right_mask is None:
+        raise ValueError("At least one of left_mask or right_mask must be provided (not None)")
+
     # To reduce the number of cases, we ensure that `left` & `left_mask`
     # always come from an array, not a scalar. This is safe, since
     # A | B == B | A
@@ -101,6 +105,10 @@ def kleene_xor(
         The result of the logical xor, and the new mask.
     """
+    # Validate precondition: at least one mask must be provided
+    if left_mask is None and right_mask is None:
+        raise ValueError("At least one of left_mask or right_mask must be provided (not None)")
+
     # To reduce the number of cases, we ensure that `left` & `left_mask`
     # always come from an array, not a scalar. This is safe, since
     # A ^ B == B ^ A
@@ -151,6 +159,10 @@ def kleene_and(
         The result of the logical xor, and the new mask.
     """
+    # Validate precondition: at least one mask must be provided
+    if left_mask is None and right_mask is None:
+        raise ValueError("At least one of left_mask or right_mask must be provided (not None)")
+
     # To reduce the number of cases, we ensure that `left` & `left_mask`
     # always come from an array, not a scalar. This is safe, since
     # A & B == B & A
```