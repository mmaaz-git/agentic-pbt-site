# Bug Report: llm cosine_similarity Division by Zero

**Target**: `llm.cosine_similarity()`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `cosine_similarity()` function crashes with `ZeroDivisionError` when either input vector is all zeros, instead of handling this edge case gracefully.

## Property-Based Test

```python
@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10), min_size=1),
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10), min_size=1)
)
@settings(max_examples=500)
def test_cosine_similarity_range(a, b):
    result = llm.cosine_similarity(a, b)
    assert -1.0 <= result <= 1.0
```

**Failing input**: `a=[0.0], b=[0.0]`

## Reproducing the Bug

```python
import llm

result = llm.cosine_similarity([0.0], [0.0])
```

Output:
```
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/llm/__init__.py", line 462, in cosine_similarity
    return dot_product / (magnitude_a * magnitude_b)
ZeroDivisionError: float division by zero
```

## Why This Is A Bug

While cosine similarity is mathematically undefined for zero vectors, crashing with an unhandled exception is not user-friendly. The function is used in `llm/embeddings.py` to calculate distance scores between embeddings.

Although legitimate embeddings should never be all zeros, the function should handle this edge case gracefully by either:
1. Returning a sensible default value (e.g., 0.0 or None)
2. Raising a more informative error (e.g., `ValueError("Cosine similarity undefined for zero vectors")`)
3. Documenting the precondition that inputs must be non-zero vectors

## Fix

```diff
diff --git a/llm/__init__.py b/llm/__init__.py
index xxx..xxx 100644
--- a/llm/__init__.py
+++ b/llm/__init__.py
@@ -458,7 +458,13 @@ def cosine_similarity(a, b):
     dot_product = sum(x * y for x, y in zip(a, b))
     magnitude_a = sum(x * x for x in a) ** 0.5
     magnitude_b = sum(x * x for x in b) ** 0.5
+
+    if magnitude_a == 0 or magnitude_b == 0:
+        # Cosine similarity is undefined for zero vectors
+        # Return 0.0 as vectors are orthogonal (have no common direction)
+        return 0.0
+
     return dot_product / (magnitude_a * magnitude_b)
```