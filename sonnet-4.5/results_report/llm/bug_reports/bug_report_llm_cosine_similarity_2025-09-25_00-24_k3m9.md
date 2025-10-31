# Bug Report: llm.cosine_similarity ZeroDivisionError with Zero Vectors

**Target**: `llm.cosine_similarity`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `cosine_similarity` function crashes with `ZeroDivisionError` when either input vector contains all zeros, which is a valid mathematical input that should be handled gracefully.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import llm


@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=100)
)
def test_cosine_similarity_no_crash_with_zero_vectors(a):
    b = [0.0] * len(a)
    result = llm.cosine_similarity(a, b)
    assert isinstance(result, (int, float))
```

**Failing input**: `a=[1.0, 2.0, 3.0], b=[0.0, 0.0, 0.0]` (and many other inputs involving zero vectors)

## Reproducing the Bug

```python
import llm

a = [1.0, 2.0, 3.0]
b = [0.0, 0.0, 0.0]

result = llm.cosine_similarity(a, b)
```

**Output:**
```
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "llm/__init__.py", line 462, in cosine_similarity
    return dot_product / (magnitude_a * magnitude_b)
ZeroDivisionError: float division by zero
```

## Why This Is A Bug

Zero vectors are mathematically valid inputs to cosine similarity. The function should either:
1. Return a defined value (e.g., 0.0 or NaN) when one or both vectors are zero
2. Raise a more informative exception (e.g., `ValueError` with a clear message)

The current behavior of crashing with `ZeroDivisionError` violates the principle of robustness - the function doesn't validate its inputs and fails with an unhelpful error message for valid mathematical inputs.

## Fix

```diff
def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = sum(x * x for x in a) ** 0.5
    magnitude_b = sum(x * x for x in b) ** 0.5
+
+   if magnitude_a == 0.0 or magnitude_b == 0.0:
+       return 0.0
+
    return dot_product / (magnitude_a * magnitude_b)
```

**Note**: The mathematical convention for cosine similarity with zero vectors varies. Returning 0.0 is one reasonable choice (treating zero vectors as orthogonal to everything). Alternatively, the function could raise a `ValueError` with a clear message, or return `float('nan')`.