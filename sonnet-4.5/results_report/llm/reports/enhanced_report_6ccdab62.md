# Bug Report: llm cosine_similarity Division by Zero with Zero Vectors

**Target**: `llm.cosine_similarity()`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `cosine_similarity()` function crashes with `ZeroDivisionError` when either input vector contains all zeros, as the magnitude calculation results in zero and causes division by zero.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import llm

@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10), min_size=1),
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10), min_size=1)
)
@settings(max_examples=500)
def test_cosine_similarity_range(a, b):
    result = llm.cosine_similarity(a, b)
    assert -1.0 <= result <= 1.0

if __name__ == "__main__":
    test_cosine_similarity_range()
```

<details>

<summary>
**Failing input**: `a=[0.0], b=[0.0]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 14, in <module>
    test_cosine_similarity_range()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 5, in test_cosine_similarity_range
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10), min_size=1),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/59/hypo.py", line 10, in test_cosine_similarity_range
    result = llm.cosine_similarity(a, b)
  File "/home/npc/miniconda/lib/python3.13/site-packages/llm/__init__.py", line 462, in cosine_similarity
    return dot_product / (magnitude_a * magnitude_b)
           ~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ZeroDivisionError: float division by zero
Falsifying example: test_cosine_similarity_range(
    # The test sometimes passed when commented parts were varied together.
    a=[0.0],  # or any other generated value
    b=[0.0],  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import llm

# Test case that should cause ZeroDivisionError
result = llm.cosine_similarity([0.0], [0.0])
print(f"Result: {result}")
```

<details>

<summary>
ZeroDivisionError: float division by zero
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/59/repo.py", line 4, in <module>
    result = llm.cosine_similarity([0.0], [0.0])
  File "/home/npc/miniconda/lib/python3.13/site-packages/llm/__init__.py", line 462, in cosine_similarity
    return dot_product / (magnitude_a * magnitude_b)
           ~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ZeroDivisionError: float division by zero
```
</details>

## Why This Is A Bug

The `cosine_similarity` function is mathematically undefined when either vector has zero magnitude (i.e., all components are zero). The current implementation at `/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/llm/__init__.py:458-462` doesn't check for this edge case before performing division, resulting in an unhandled `ZeroDivisionError`.

While zero vectors may not be common in typical embedding use cases, the function accepts arbitrary numeric lists as input and should handle all valid numeric inputs gracefully. Crashing with an unhandled exception violates the principle of defensive programming and makes the function less robust. The function should either return a sensible default value or raise a more informative error message.

## Relevant Context

The `cosine_similarity` function is a utility function in the llm library that calculates the cosine of the angle between two vectors. It's commonly used to measure similarity between embeddings or feature vectors. The function is defined in `llm/__init__.py` and exported as part of the public API.

The mathematical formula for cosine similarity is:
```
cosine_similarity(a, b) = (a · b) / (||a|| × ||b||)
```

When either ||a|| or ||b|| equals zero (zero vector), the denominator becomes zero, making the operation undefined. This is the root cause of the crash.

Documentation: The function lacks docstring documentation explaining its behavior or preconditions.
Code location: `/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/llm/__init__.py:458-462`

## Proposed Fix

```diff
--- a/llm/__init__.py
+++ b/llm/__init__.py
@@ -456,7 +456,14 @@ def decode(binary):


 def cosine_similarity(a, b):
+    """Calculate cosine similarity between two vectors.
+    Returns 0.0 for zero vectors since they have no defined direction."""
     dot_product = sum(x * y for x, y in zip(a, b))
     magnitude_a = sum(x * x for x in a) ** 0.5
     magnitude_b = sum(x * x for x in b) ** 0.5
+
+    if magnitude_a == 0 or magnitude_b == 0:
+        # Cosine similarity is undefined for zero vectors
+        # Return 0.0 as a sensible default (orthogonal/no similarity)
+        return 0.0
+
     return dot_product / (magnitude_a * magnitude_b)
```