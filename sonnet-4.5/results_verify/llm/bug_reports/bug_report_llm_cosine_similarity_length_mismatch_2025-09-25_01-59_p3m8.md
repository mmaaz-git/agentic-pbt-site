# Bug Report: llm.cosine_similarity Silent Mismatched Length Handling

**Target**: `llm.cosine_similarity`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `cosine_similarity` function silently produces mathematically incorrect results when given vectors of different lengths, due to using `zip()` which truncates to the shorter length without warning.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st, assume
import llm

@settings(max_examples=500)
@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=1, max_size=10),
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=1, max_size=10)
)
def test_cosine_similarity_length_check(a, b):
    if len(a) != len(b):
        try:
            result = llm.cosine_similarity(a, b)
            assert False, (
                f"cosine_similarity should reject mismatched lengths but returned {result} "
                f"for a (len={len(a)}) and b (len={len(b)})"
            )
        except (ValueError, AssertionError):
            pass
```

**Failing input**: `a=[1, 0]`, `b=[1]`

## Reproducing the Bug

```python
import llm

a = [1, 0]
b = [1]
result = llm.cosine_similarity(a, b)

print(f"cosine_similarity({a}, {b}) = {result}")
```

**Output:**
```
cosine_similarity([1, 0], [1]) = 1.0
```

This is mathematically incorrect. The function computes:
- `dot_product = 1*1 = 1` (zip truncates, ignoring the second element of `a`)
- `magnitude_a = sqrt(1² + 0²) = 1.0`
- `magnitude_b = sqrt(1²) = 1.0`
- `result = 1 / (1 * 1) = 1.0`

The true cosine similarity is undefined for vectors of different dimensionality.

## Why This Is A Bug

Cosine similarity is only mathematically defined for vectors of the same dimensionality. The function's use of `zip()` silently truncates the longer vector and computes a result using partial data, which is incorrect and misleading.

This bug can lead to:
1. **Silent data corruption**: Callers receive a numeric result that appears valid but is mathematically wrong
2. **Incorrect embeddings comparison**: In ML applications, this could silently compare truncated embeddings
3. **Hard-to-debug issues**: The function doesn't warn users that inputs were invalid

The bug is particularly insidious because it doesn't crash—it returns plausible-looking floating point values that hide the underlying error.

## Fix

```diff
--- a/llm/__init__.py
+++ b/llm/__init__.py
@@ -457,6 +457,10 @@ def decode(binary):

 def cosine_similarity(a, b):
+    if len(a) != len(b):
+        raise ValueError(
+            f"Vectors must have the same length: len(a)={len(a)}, len(b)={len(b)}"
+        )
     dot_product = sum(x * y for x, y in zip(a, b))
     magnitude_a = sum(x * x for x in a) ** 0.5
     magnitude_b = sum(x * x for x in b) ** 0.5
```

This fix validates that vectors have matching lengths before computation, raising a clear error message when they don't, preventing silent incorrect calculations.