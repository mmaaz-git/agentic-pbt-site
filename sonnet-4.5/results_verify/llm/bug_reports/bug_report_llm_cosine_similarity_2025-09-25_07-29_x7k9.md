# Bug Report: llm.cosine_similarity - Division by Zero on Zero Vectors

**Target**: `llm.cosine_similarity`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `cosine_similarity` function crashes with `ZeroDivisionError` when either input vector has zero magnitude, making it unable to handle valid mathematical inputs.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import llm

@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1),
    st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1)
)
def test_cosine_similarity_no_crash(a, b):
    if len(a) != len(b):
        return
    llm.cosine_similarity(a, b)
```

**Failing input**: `a=[0.0, 0.0], b=[1.0, 2.0]`

## Reproducing the Bug

```python
import llm

result = llm.cosine_similarity([0.0, 0.0], [1.0, 2.0])
```

**Output:**
```
ZeroDivisionError: float division by zero
```

## Why This Is A Bug

The cosine similarity function should handle zero vectors gracefully. In mathematics, cosine similarity with a zero vector is undefined, but the function should either return a sentinel value (like 0.0 or NaN) or raise a more descriptive error, not crash with `ZeroDivisionError`. Zero vectors are valid inputs that users might encounter with real data.

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