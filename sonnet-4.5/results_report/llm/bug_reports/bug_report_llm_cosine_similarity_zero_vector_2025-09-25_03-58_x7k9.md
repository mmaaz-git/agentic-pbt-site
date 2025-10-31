# Bug Report: llm.cosine_similarity Crashes on Zero Vectors

**Target**: `llm.cosine_similarity`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `cosine_similarity` function crashes with a `ZeroDivisionError` when either input vector has zero magnitude, instead of handling this edge case gracefully.

## Property-Based Test

```python
import llm
from hypothesis import given, strategies as st
import math

@given(st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100), min_size=1, max_size=10))
def test_cosine_similarity_handles_zero_vectors(b):
    a = [0.0] * len(b)

    try:
        result = llm.cosine_similarity(a, b)
        assert False, f"cosine_similarity with zero vector returned {result}, should raise ValueError"
    except ZeroDivisionError:
        assert False, "Should raise ValueError, not ZeroDivisionError"
    except ValueError:
        pass
```

**Failing input**: `a = [0.0, 0.0, 0.0]`, `b = [1.0, 2.0, 3.0]`

## Reproducing the Bug

```python
import llm

a = [0.0, 0.0, 0.0]
b = [1.0, 2.0, 3.0]

try:
    result = llm.cosine_similarity(a, b)
    print(f"Result: {result}")
except ZeroDivisionError as e:
    print(f"ZeroDivisionError: {e}")
```

Output:
```
ZeroDivisionError: float division by zero
```

## Why This Is A Bug

Cosine similarity is mathematically undefined for zero-magnitude vectors. The function should detect this case and raise a descriptive `ValueError` explaining the issue, rather than crashing with a cryptic `ZeroDivisionError`.

This affects users who:
1. Pass input data that might contain zero vectors
2. Use embeddings or vector representations that could be all zeros in edge cases
3. Need clear error messages to debug their code

## Fix

```diff
def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = sum(x * x for x in a) ** 0.5
    magnitude_b = sum(x * x for x in b) ** 0.5
+
+   if magnitude_a == 0 or magnitude_b == 0:
+       raise ValueError(
+           "Cannot compute cosine similarity with zero-magnitude vector(s). "
+           f"magnitude(a)={magnitude_a}, magnitude(b)={magnitude_b}"
+       )
+
    return dot_product / (magnitude_a * magnitude_b)
```