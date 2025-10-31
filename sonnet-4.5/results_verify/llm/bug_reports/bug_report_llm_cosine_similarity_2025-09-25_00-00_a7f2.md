# Bug Report: llm.cosine_similarity Division by Zero

**Target**: `llm.cosine_similarity`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `cosine_similarity` function crashes with `ZeroDivisionError` when either input vector has zero magnitude, which is valid input that should be handled gracefully.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
import llm

@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1),
    st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1)
)
def test_cosine_similarity_handles_zero_vectors(a, b):
    assume(len(a) == len(b))
    result = llm.cosine_similarity(a, b)
    assert -1.0 <= result <= 1.0 or result != result  # NaN is acceptable
```

**Failing input**: `a = [0.0, 0.0, 0.0], b = [1.0, 2.0, 3.0]`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

import llm

a = [0.0, 0.0, 0.0]
b = [1.0, 2.0, 3.0]

result = llm.cosine_similarity(a, b)
```

Output:
```
ZeroDivisionError: float division by zero
```

## Why This Is A Bug

1. Zero vectors are mathematically valid input
2. The function has no documented precondition requiring non-zero magnitude
3. The cosine similarity between a zero vector and any other vector is undefined (not an error case to crash on)
4. Industry-standard implementations (e.g., scipy, numpy) return NaN or 0.0 for this case rather than crashing

## Fix

Handle zero-magnitude vectors by checking before division:

```diff
def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = sum(x * x for x in a) ** 0.5
    magnitude_b = sum(x * x for x in b) ** 0.5
+   if magnitude_a == 0 or magnitude_b == 0:
+       return 0.0  # or float('nan') depending on desired behavior
    return dot_product / (magnitude_a * magnitude_b)
```