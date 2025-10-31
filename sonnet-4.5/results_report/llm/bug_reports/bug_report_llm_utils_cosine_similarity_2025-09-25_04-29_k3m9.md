# Bug Report: llm.utils.cosine_similarity Division by Zero

**Target**: `llm.utils.cosine_similarity`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `cosine_similarity` function crashes with a `ZeroDivisionError` when either input vector has zero magnitude (all zeros or empty).

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
import llm.utils as utils

@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=1),
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=1)
)
def test_cosine_similarity_handles_zero_vectors(a, b):
    assume(len(a) == len(b))
    try:
        result = utils.cosine_similarity(a, b)
        if sum(x * x for x in a) > 0 and sum(x * x for x in b) > 0:
            assert -1 <= result <= 1
    except ZeroDivisionError:
        assert False, "Should handle zero vectors gracefully"
```

**Failing input**: `a = [0.0, 0.0]`, `b = [1.0, 2.0]`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.utils import cosine_similarity

a = [0.0, 0.0, 0.0]
b = [1.0, 2.0, 3.0]

result = cosine_similarity(a, b)
```

**Output:**
```
ZeroDivisionError: float division by zero
```

## Why This Is A Bug

The cosine similarity is mathematically undefined for zero vectors, but the function should either:
1. Return a special value (like 0.0 or NaN)
2. Raise a descriptive ValueError instead of letting ZeroDivisionError propagate

Valid inputs (lists of numbers) should not cause unhandled exceptions. Users might encounter zero vectors in real data (e.g., sparse embeddings, initialization values).

## Fix

```diff
--- a/llm/utils.py
+++ b/llm/utils.py
@@ -458,6 +458,10 @@ def cosine_similarity(a, b):
     dot_product = sum(x * y for x, y in zip(a, b))
     magnitude_a = sum(x * x for x in a) ** 0.5
     magnitude_b = sum(x * x for x in b) ** 0.5
+
+    if magnitude_a == 0.0 or magnitude_b == 0.0:
+        return 0.0  # or raise ValueError("Cosine similarity undefined for zero vectors")
+
     return dot_product / (magnitude_a * magnitude_b)
```