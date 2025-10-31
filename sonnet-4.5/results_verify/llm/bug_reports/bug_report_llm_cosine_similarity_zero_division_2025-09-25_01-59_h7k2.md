# Bug Report: llm.cosine_similarity Division by Zero

**Target**: `llm.cosine_similarity`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `cosine_similarity` function crashes with a `ZeroDivisionError` when either input vector contains all zeros or is empty, which are valid inputs that should be handled gracefully.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
import llm

@settings(max_examples=500)
@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=1),
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=1)
)
def test_cosine_similarity_no_crash(a, b):
    try:
        result = llm.cosine_similarity(a, b)
        assert isinstance(result, (int, float))
    except ZeroDivisionError:
        assert False, f"cosine_similarity crashed with ZeroDivisionError on inputs a={a}, b={b}"
```

**Failing input**: `a=[0.0, 0.0]`, `b=[0.0, 0.0]`

## Reproducing the Bug

```python
import llm

result = llm.cosine_similarity([0, 0], [0, 0])
```

**Output:**
```
ZeroDivisionError: float division by zero
```

Additional failing examples:
- `llm.cosine_similarity([1, 2, 3], [0, 0, 0])` → ZeroDivisionError
- `llm.cosine_similarity([0], [0])` → ZeroDivisionError
- `llm.cosine_similarity([], [])` → ZeroDivisionError (empty vectors)

## Why This Is A Bug

Zero vectors are mathematically valid inputs for embedding similarity calculations. While the cosine similarity is technically undefined for zero vectors (since you cannot normalize a zero-length vector), the function should either:

1. Return a sensible default value (e.g., 0.0 or NaN)
2. Raise a descriptive error message explaining the mathematical constraint

Currently, the function raises an unhelpful `ZeroDivisionError` that doesn't explain why the operation failed. This is particularly problematic in production systems where zero vectors might arise from data processing, empty documents, or masked embeddings.

## Fix

```diff
--- a/llm/__init__.py
+++ b/llm/__init__.py
@@ -458,7 +458,13 @@ def cosine_similarity(a, b):
     dot_product = sum(x * y for x, y in zip(a, b))
     magnitude_a = sum(x * x for x in a) ** 0.5
     magnitude_b = sum(x * x for x in b) ** 0.5
+
+    # Handle zero vectors
+    if magnitude_a == 0 or magnitude_b == 0:
+        return 0.0
+
     return dot_product / (magnitude_a * magnitude_b)
```

This fix returns 0.0 for zero vectors, which is a common convention in vector similarity calculations and aligns with the behavior of libraries like scikit-learn's cosine_similarity function.