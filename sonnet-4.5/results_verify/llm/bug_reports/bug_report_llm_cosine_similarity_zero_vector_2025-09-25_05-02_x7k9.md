# Bug Report: llm.cosine_similarity Zero Vector Division Error

**Target**: `llm.cosine_similarity`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `cosine_similarity` function crashes with a `ZeroDivisionError` when either input vector is a zero vector (all elements are zero), instead of handling this edge case gracefully.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
import llm

@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=1, max_size=100),
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=1, max_size=100),
)
def test_cosine_similarity_handles_zero_vectors(a, b):
    assume(len(a) == len(b))
    try:
        result = llm.cosine_similarity(a, b)
        assert isinstance(result, float)
    except ZeroDivisionError:
        pytest.fail("cosine_similarity should handle zero vectors gracefully")
```

**Failing input**: `a=[0.0, 0.0, 0.0], b=[1.0, 2.0, 3.0]`

## Reproducing the Bug

```python
import llm

a = [0.0, 0.0, 0.0]
b = [1.0, 2.0, 3.0]

result = llm.cosine_similarity(a, b)
```

**Output:**
```
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "llm/__init__.py", line 462, in cosine_similarity
    return dot_product / (magnitude_a * magnitude_b)
           ~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~
ZeroDivisionError: float division by zero
```

## Why This Is A Bug

Cosine similarity is mathematically undefined for zero vectors (vectors with magnitude 0), but the function should handle this edge case gracefully rather than crashing. This can occur in real-world scenarios with embeddings or other vector operations where some vectors might be all zeros.

The bug occurs at line 462 in `__init__.py`:
```python
def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = sum(x * x for x in a) ** 0.5
    magnitude_b = sum(x * x for x in b) ** 0.5
    return dot_product / (magnitude_a * magnitude_b)  # Crashes here if either magnitude is 0
```

When either vector is all zeros, its magnitude becomes 0, causing division by zero.

## Fix

Handle zero vectors explicitly by checking for zero magnitudes:

```diff
--- a/llm/__init__.py
+++ b/llm/__init__.py
@@ -458,7 +458,11 @@ def cosine_similarity(a, b):
     dot_product = sum(x * y for x, y in zip(a, b))
     magnitude_a = sum(x * x for x in a) ** 0.5
     magnitude_b = sum(x * x for x in b) ** 0.5
-    return dot_product / (magnitude_a * magnitude_b)
+    denominator = magnitude_a * magnitude_b
+    if denominator == 0:
+        # Return 0 for zero vectors (similarity undefined, but 0 is conventional)
+        return 0.0
+    return dot_product / denominator
```

Alternative fix using epsilon to avoid exact zero comparison:

```diff
--- a/llm/__init__.py
+++ b/llm/__init__.py
@@ -458,7 +458,12 @@ def cosine_similarity(a, b):
     dot_product = sum(x * y for x, y in zip(a, b))
     magnitude_a = sum(x * x for x in a) ** 0.5
     magnitude_b = sum(x * x for x in b) ** 0.5
-    return dot_product / (magnitude_a * magnitude_b)
+    denominator = magnitude_a * magnitude_b
+    if denominator < 1e-10:
+        # Vectors too small or zero - return 0
+        return 0.0
+    return dot_product / denominator
```