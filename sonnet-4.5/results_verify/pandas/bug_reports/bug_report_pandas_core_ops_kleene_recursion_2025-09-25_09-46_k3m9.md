# Bug Report: pandas.core.ops Kleene Operations RecursionError

**Target**: `pandas.core.ops.kleene_and`, `pandas.core.ops.kleene_or`, `pandas.core.ops.kleene_xor`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When both `left_mask` and `right_mask` are `None`, the Kleene logic functions (`kleene_and`, `kleene_or`, `kleene_xor`) enter infinite recursion and crash with `RecursionError`. According to the docstring, "Only one of these may be None", but the functions don't validate this precondition and instead infinitely swap arguments.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
from pandas.core.ops import kleene_and, kleene_or, kleene_xor
from pandas._libs import missing as libmissing

bool_or_na = st.one_of(st.booleans(), st.just(libmissing.NA))

@given(left=bool_or_na, right=bool_or_na)
@settings(max_examples=500)
def test_kleene_and_commutativity_scalars(left, right):
    result1, mask1 = kleene_and(left, right, None, None)
    result2, mask2 = kleene_and(right, left, None, None)
    assert result1 == result2
    assert mask1 == mask2
```

**Failing input**: `left=False, right=False` (or any combination of two scalars)

## Reproducing the Bug

```python
from pandas.core.ops import kleene_and, kleene_or, kleene_xor

result, mask = kleene_and(False, False, None, None)
```

Output:
```
RecursionError: maximum recursion depth exceeded
```

All three functions (`kleene_and`, `kleene_or`, `kleene_xor`) exhibit the same bug:

```python
from pandas.core.ops import kleene_and, kleene_or, kleene_xor

kleene_and(True, False, None, None)
kleene_or(True, False, None, None)
kleene_xor(True, False, None, None)
```

## Why This Is A Bug

The docstring states: "Only one of these may be None, which implies that the associated `left` or `right` value is a scalar." This is a documented precondition.

However, when this precondition is violated:
1. The function should raise a clear `ValueError` explaining the invalid input
2. Instead, it enters infinite recursion because each function checks `if left_mask is None:` and swaps arguments
3. When both masks are None, it keeps swapping forever: `kleene_and(left, right, None, None)` → `kleene_and(right, left, None, None)` → `kleene_and(left, right, None, None)` → ...
4. This eventually crashes with `RecursionError`, which is extremely unfriendly and hard to debug

## Fix

```diff
--- a/pandas/core/ops/mask_ops.py
+++ b/pandas/core/ops/mask_ops.py
@@ -154,6 +154,10 @@ def kleene_and(
     result, mask: ndarray[bool]
         The result of the logical xor, and the new mask.
     """
+    if left_mask is None and right_mask is None:
+        raise ValueError(
+            "At least one of left_mask or right_mask must be provided"
+        )
     # To reduce the number of cases, we ensure that `left` & `left_mask`
     # always come from an array, not a scalar. This is safe, since
     # A & B == B & A
     if left_mask is None:
```

Apply the same fix to `kleene_or` and `kleene_xor`.