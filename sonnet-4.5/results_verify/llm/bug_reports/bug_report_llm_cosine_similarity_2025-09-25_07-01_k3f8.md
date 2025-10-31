# Bug Report: llm.cosine_similarity Division By Zero

**Target**: `llm.cosine_similarity`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `cosine_similarity` function crashes with a `ZeroDivisionError` when given vectors with very small magnitudes that underflow to zero during computation.

## Property-Based Test

```python
from hypothesis import assume, given, strategies as st
import llm

@given(st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=1))
def test_cosine_similarity_self(a):
    assume(any(x != 0 for x in a))
    result = llm.cosine_similarity(a, a)
    assert result == 1.0
```

**Failing input**: `a=[2.225073858507203e-309]`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, "/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages")

import llm

result = llm.cosine_similarity([2.225073858507203e-309], [2.225073858507203e-309])
```

Output:
```
ZeroDivisionError: float division by zero
```

## Why This Is A Bug

The function computes magnitudes using `sum(x * x for x in a) ** 0.5`, but when vector elements are very small (near the floating-point underflow limit), the squared values underflow to exactly 0.0, making the magnitude 0.0 and causing division by zero. This is valid input (non-NaN, non-infinite floats), so the function should handle it gracefully.

## Fix

```diff
--- a/llm/__init__.py
+++ b/llm/__init__.py
@@ -458,7 +458,11 @@ def cosine_similarity(a, b):
 def cosine_similarity(a, b):
     dot_product = sum(x * y for x, y in zip(a, b))
     magnitude_a = sum(x * x for x in a) ** 0.5
     magnitude_b = sum(x * x for x in b) ** 0.5
-    return dot_product / (magnitude_a * magnitude_b)
+    denominator = magnitude_a * magnitude_b
+    if denominator == 0:
+        # Return 0 for zero vectors (undefined cosine similarity)
+        return 0.0
+    return dot_product / denominator
```