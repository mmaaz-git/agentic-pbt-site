# Bug Report: llm.cosine_similarity Zero Vector Division

**Target**: `llm.cosine_similarity`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `cosine_similarity` function crashes with `ZeroDivisionError` when either input vector contains all zeros, which is a valid input that should be handled gracefully.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume

@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=1),
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=1)
)
def test_cosine_similarity_no_crash(a, b):
    assume(len(a) == len(b))
    result = llm.cosine_similarity(a, b)
    assert isinstance(result, float)
```

**Failing input**: `a = [0.0, 0.0, 0.0]`, `b = [1.0, 2.0, 3.0]`

## Reproducing the Bug

```python
import llm

a = [0.0, 0.0, 0.0]
b = [1.0, 2.0, 3.0]

result = llm.cosine_similarity(a, b)
```

**Output**:
```
ZeroDivisionError: float division by zero
```

## Why This Is A Bug

The cosine similarity is mathematically undefined when one or both vectors are zero vectors (magnitude = 0). However, the function should not crash with an unhandled exception. Instead, it should either:

1. Return a special value (0, nan, or None) to indicate undefined similarity
2. Raise a descriptive ValueError explaining that zero vectors are not supported
3. Document in the docstring that zero vectors are not allowed

Currently, the function has no docstring and provides no guidance on this edge case, making it appear as if any list of numbers is valid input.

## Fix

```diff
--- a/llm/__init__.py
+++ b/llm/__init__.py
@@ -458,6 +458,10 @@ def cosine_similarity(a, b):
     dot_product = sum(x * y for x, y in zip(a, b))
     magnitude_a = sum(x * x for x in a) ** 0.5
     magnitude_b = sum(x * x for x in b) ** 0.5
+
+    if magnitude_a == 0 or magnitude_b == 0:
+        return 0.0
+
     return dot_product / (magnitude_a * magnitude_b)
```

Alternative fix with explicit error:
```diff
--- a/llm/__init__.py
+++ b/llm/__init__.py
@@ -458,6 +458,10 @@ def cosine_similarity(a, b):
     dot_product = sum(x * y for x, y in zip(a, b))
     magnitude_a = sum(x * x for x in a) ** 0.5
     magnitude_b = sum(x * x for x in b) ** 0.5
+
+    if magnitude_a == 0 or magnitude_b == 0:
+        raise ValueError("cosine_similarity requires non-zero vectors")
+
     return dot_product / (magnitude_a * magnitude_b)
```