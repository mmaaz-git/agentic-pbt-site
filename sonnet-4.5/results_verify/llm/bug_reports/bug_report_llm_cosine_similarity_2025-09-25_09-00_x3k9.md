# Bug Report: llm.cosine_similarity Division by Zero and Incorrect Results

**Target**: `llm.cosine_similarity`
**Severity**: High
**Bug Type**: Crash, Logic
**Date**: 2025-09-25

## Summary

The `cosine_similarity` function has two critical bugs: (1) it crashes with `ZeroDivisionError` when either input vector has magnitude 0, and (2) it produces mathematically incorrect results when input vectors have different lengths due to mismatched iteration (using `zip`) for dot product vs full iteration for magnitudes.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import llm

@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=1),
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=1)
)
def test_cosine_similarity_no_crash(a, b):
    if len(a) != len(b):
        return
    result = llm.cosine_similarity(a, b)
    assert -1.0 <= result <= 1.0
```

**Failing input for bug 1**: `a = [0, 0], b = [1, 1]` (zero vector)
**Failing input for bug 2**: `a = [1, 2, 3], b = [4, 5]` (different lengths)

## Reproducing the Bug

```python
import llm

print("Bug 1: Division by zero")
llm.cosine_similarity([0, 0], [1, 1])

print("\nBug 2: Incorrect result for different lengths")
result = llm.cosine_similarity([1, 2, 3], [4, 5])
print(f"Result: {result}")

dot_product_correct = 1*4 + 2*5 + 3*0
magnitude_a = (1**2 + 2**2 + 3**2)**0.5
magnitude_b = (4**2 + 5**2)**0.5
correct = dot_product_correct / (magnitude_a * magnitude_b)
print(f"Expected (zero-padding): {correct}")
print(f"Got: {result}")
```

## Why This Is A Bug

**Bug 1**: Cosine similarity is mathematically undefined for zero vectors (vectors with magnitude 0). The function should either return a special value (e.g., `0.0` or `None`) or raise a descriptive error, not crash with `ZeroDivisionError`.

**Bug 2**: When vectors have different lengths, `zip(a, b)` truncates to the shorter length for the dot product calculation, but the magnitude calculations use the full length of each vector. This produces mathematically incorrect results. For example:
- `a = [1, 2, 3]`, `b = [4, 5]`
- `dot_product = 1*4 + 2*5 = 14` (only 2 terms due to zip)
- `magnitude_a = sqrt(1² + 2² + 3²) = sqrt(14)` (3 terms)
- `magnitude_b = sqrt(4² + 5²) = sqrt(41)` (2 terms)
- Result: `14 / (sqrt(14) * sqrt(41))` ≈ `0.584`

The correct behavior should either:
1. Require vectors to have the same length (raise `ValueError`)
2. Zero-pad the shorter vector to match lengths

## Fix

```diff
--- a/llm/__init__.py
+++ b/llm/__init__.py
@@ -457,6 +457,15 @@ def decode(binary):


 def cosine_similarity(a, b):
+    if len(a) != len(b):
+        raise ValueError(
+            f"Vectors must have the same length. Got {len(a)} and {len(b)}"
+        )
+    magnitude_a = sum(x * x for x in a) ** 0.5
+    magnitude_b = sum(x * x for x in b) ** 0.5
+    if magnitude_a == 0 or magnitude_b == 0:
+        return 0.0
     dot_product = sum(x * y for x, y in zip(a, b))
-    magnitude_a = sum(x * x for x in a) ** 0.5
-    magnitude_b = sum(x * x for x in b) ** 0.5
     return dot_product / (magnitude_a * magnitude_b)
```