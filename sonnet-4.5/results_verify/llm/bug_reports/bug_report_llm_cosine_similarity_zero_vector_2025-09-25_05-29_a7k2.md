# Bug Report: llm.cosine_similarity Division by Zero with Zero Vectors

**Target**: `llm.cosine_similarity`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `cosine_similarity` function crashes with a `ZeroDivisionError` when either input vector is the zero vector, instead of handling this edge case gracefully.

## Property-Based Test

```python
from hypothesis import given, assume, strategies as st
import llm

@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10), min_size=1),
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10), min_size=1)
)
def test_cosine_similarity_no_crash(a, b):
    result = llm.cosine_similarity(a, b)
```

**Failing input**: `a=[0.0, 0.0]`, `b=[1.0, 2.0]`

## Reproducing the Bug

```python
import llm

result = llm.cosine_similarity([0.0, 0.0], [1.0, 2.0])
```

Output:
```
Traceback (most recent call last):
  ...
ZeroDivisionError: float division by zero
```

## Why This Is A Bug

Cosine similarity is commonly used in embeddings and vector operations. The zero vector is a valid mathematical input, and the function should either:
1. Return a mathematically sensible value (e.g., 0.0 or NaN)
2. Raise a descriptive `ValueError` explaining that zero vectors are not supported

Instead, it raises a cryptic `ZeroDivisionError`, which is poor error handling. The mathematical definition of cosine similarity is undefined for zero vectors, so this should be explicitly handled.

## Fix

```diff
--- a/llm/__init__.py
+++ b/llm/__init__.py
@@ -458,6 +458,10 @@ def decode(binary):
 def cosine_similarity(a, b):
     dot_product = sum(x * y for x, y in zip(a, b))
     magnitude_a = sum(x * x for x in a) ** 0.5
     magnitude_b = sum(x * x for x in b) ** 0.5
+    if magnitude_a == 0 or magnitude_b == 0:
+        raise ValueError(
+            "cosine_similarity is undefined for zero vectors"
+        )
     return dot_product / (magnitude_a * magnitude_b)
```