# Bug Report: pandas.core.ops kleene_and/kleene_or/kleene_xor Infinite Recursion with Scalar Arguments

**Target**: `pandas.core.ops.kleene_and`, `pandas.core.ops.kleene_or`, `pandas.core.ops.kleene_xor`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The kleene logic functions in pandas.core.ops enter infinite recursion and crash with RecursionError when called with two scalar boolean arguments (both masks as None), violating their documented constraint that "only one of these may be None."

## Property-Based Test

```python
from pandas.core import ops
from hypothesis import given, strategies as st

@given(st.booleans(), st.booleans())
def test_kleene_and_without_mask_equals_regular_and(a, b):
    result = ops.kleene_and(a, b, None, None)
    expected = a and b
    assert result == expected

if __name__ == "__main__":
    test_kleene_and_without_mask_equals_regular_and()
```

<details>

<summary>
**Failing input**: `a=False, b=True` (also fails with `a=False, b=False`)
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 11, in <module>
  |     test_kleene_and_without_mask_equals_regular_and()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 5, in test_kleene_and_without_mask_equals_regular_and
  |     def test_kleene_and_without_mask_equals_regular_and(a, b):
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 6, in test_kleene_and_without_mask_equals_regular_and
    |     result = ops.kleene_and(a, b, None, None)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/ops/mask_ops.py", line 157, in kleene_and
    |     return kleene_and(right, left, right_mask, left_mask)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/ops/mask_ops.py", line 157, in kleene_and
    |     return kleene_and(right, left, right_mask, left_mask)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/ops/mask_ops.py", line 157, in kleene_and
    |     return kleene_and(right, left, right_mask, left_mask)
    |   [Previous line repeated 1997 more times]
    | RecursionError: maximum recursion depth exceeded
    | Falsifying example: test_kleene_and_without_mask_equals_regular_and(
    |     a=False,
    |     b=True,
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 6, in test_kleene_and_without_mask_equals_regular_and
    |     result = ops.kleene_and(a, b, None, None)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/ops/mask_ops.py", line 157, in kleene_and
    |     return kleene_and(right, left, right_mask, left_mask)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/ops/mask_ops.py", line 157, in kleene_and
    |     return kleene_and(right, left, right_mask, left_mask)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/ops/mask_ops.py", line 157, in kleene_and
    |     return kleene_and(right, left, right_mask, left_mask)
    |   [Previous line repeated 1997 more times]
    | RecursionError: maximum recursion depth exceeded
    | Falsifying example: test_kleene_and_without_mask_equals_regular_and(
    |     a=False,
    |     b=False,
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
from pandas.core import ops

# Test case that causes RecursionError
result = ops.kleene_and(False, True, None, None)
print(f"Result: {result}")
```

<details>

<summary>
RecursionError: maximum recursion depth exceeded
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/40/repo.py", line 4, in <module>
    result = ops.kleene_and(False, True, None, None)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/ops/mask_ops.py", line 157, in kleene_and
    return kleene_and(right, left, right_mask, left_mask)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/ops/mask_ops.py", line 157, in kleene_and
    return kleene_and(right, left, right_mask, left_mask)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/ops/mask_ops.py", line 157, in kleene_and
    return kleene_and(right, left, right_mask, left_mask)
  [Previous line repeated 996 more times]
RecursionError: maximum recursion depth exceeded
```
</details>

## Why This Is A Bug

The functions crash with infinite recursion when both mask arguments are None, even though this violates their documented constraint. The issue occurs in the argument-swapping logic designed to ensure the left operand is always an array. When both masks are None (indicating both operands are scalars), the swap operation creates an infinite loop:

1. The function docstrings explicitly state: "Only one of these may be None, which implies that the associated `left` or `right` value is a scalar."
2. The function signatures accept `bool` types: `left, right : ndarray, NA, or bool`
3. When `left_mask is None`, the functions swap arguments and recursively call themselves
4. If both masks are None, after swapping, `left_mask` remains None (it was `right_mask`), triggering another swap ad infinitum
5. All three functions (`kleene_and`, `kleene_or`, `kleene_xor`) exhibit the same bug at lines 157, 43, and 107 respectively

While the documentation states this input combination is invalid, the functions should handle this gracefully with a clear error message rather than crashing with RecursionError.

## Relevant Context

- These functions are exposed in `pandas.core.ops.__all__` making them accessible via `pandas.core.ops.kleene_and/or/xor`
- The module is marked as "not a public API" in `/pandas/core/ops/__init__.py:4`
- The functions are primarily internal utilities used by `BooleanArray` operations where at least one mask is always present
- In normal pandas usage, these functions work correctly as BooleanArray always provides its `_mask` attribute
- The bug only manifests when calling these functions directly with two scalar boolean values

Code locations:
- `/pandas/core/ops/mask_ops.py:156-157` (kleene_and)
- `/pandas/core/ops/mask_ops.py:42-43` (kleene_or)
- `/pandas/core/ops/mask_ops.py:106-107` (kleene_xor)

## Proposed Fix

```diff
--- a/pandas/core/ops/mask_ops.py
+++ b/pandas/core/ops/mask_ops.py
@@ -39,6 +39,10 @@ def kleene_or(
     # To reduce the number of cases, we ensure that `left` & `left_mask`
     # always come from an array, not a scalar. This is safe, since
     # A | B == B | A
+    if left_mask is None and right_mask is None:
+        raise TypeError(
+            "kleene_or requires at least one array argument (both masks cannot be None)"
+        )
     if left_mask is None:
         return kleene_or(right, left, right_mask, left_mask)

@@ -103,6 +107,10 @@ def kleene_xor(
     # To reduce the number of cases, we ensure that `left` & `left_mask`
     # always come from an array, not a scalar. This is safe, since
     # A ^ B == B ^ A
+    if left_mask is None and right_mask is None:
+        raise TypeError(
+            "kleene_xor requires at least one array argument (both masks cannot be None)"
+        )
     if left_mask is None:
         return kleene_xor(right, left, right_mask, left_mask)

@@ -153,6 +161,10 @@ def kleene_and(
     # To reduce the number of cases, we ensure that `left` & `left_mask`
     # always come from an array, not a scalar. This is safe, since
     # A & B == B & A
+    if left_mask is None and right_mask is None:
+        raise TypeError(
+            "kleene_and requires at least one array argument (both masks cannot be None)"
+        )
     if left_mask is None:
         return kleene_and(right, left, right_mask, left_mask)
```