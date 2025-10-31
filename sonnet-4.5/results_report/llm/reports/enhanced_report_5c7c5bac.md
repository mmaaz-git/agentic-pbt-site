# Bug Report: llm.cosine_similarity Division by Zero with Zero Vectors

**Target**: `llm.cosine_similarity`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `cosine_similarity` function crashes with a `ZeroDivisionError` when either input vector is the zero vector, instead of providing a descriptive error message about the mathematical impossibility of calculating cosine similarity with zero vectors.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import llm

@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10), min_size=1),
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10), min_size=1)
)
def test_cosine_similarity_no_crash(a, b):
    result = llm.cosine_similarity(a, b)

# Run the test
test_cosine_similarity_no_crash()
```

<details>

<summary>
**Failing input**: `a=[0.0], b=[0.0]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/9/hypo.py", line 12, in <module>
    test_cosine_similarity_no_crash()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/9/hypo.py", line 5, in test_cosine_similarity_no_crash
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10), min_size=1),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/9/hypo.py", line 9, in test_cosine_similarity_no_crash
    result = llm.cosine_similarity(a, b)
  File "/home/npc/miniconda/lib/python3.13/site-packages/llm/__init__.py", line 462, in cosine_similarity
    return dot_product / (magnitude_a * magnitude_b)
           ~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ZeroDivisionError: float division by zero
Falsifying example: test_cosine_similarity_no_crash(
    # The test sometimes passed when commented parts were varied together.
    a=[0.0],  # or any other generated value
    b=[0.0],  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import llm

# Test case that crashes with zero vector
result = llm.cosine_similarity([0.0, 0.0], [1.0, 2.0])
print(f"Result: {result}")
```

<details>

<summary>
ZeroDivisionError: float division by zero
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/9/repo.py", line 4, in <module>
    result = llm.cosine_similarity([0.0, 0.0], [1.0, 2.0])
  File "/home/npc/miniconda/lib/python3.13/site-packages/llm/__init__.py", line 462, in cosine_similarity
    return dot_product / (magnitude_a * magnitude_b)
           ~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ZeroDivisionError: float division by zero
```
</details>

## Why This Is A Bug

The function crashes with a generic `ZeroDivisionError` that provides no context about the actual problem: cosine similarity is mathematically undefined for zero vectors. The error occurs because:

1. **Mathematical Definition**: Cosine similarity is defined as cos(θ) = (A · B) / (||A|| ||B||), where ||A|| and ||B|| are the magnitudes of vectors A and B. When either vector is zero, its magnitude is 0, causing division by zero.

2. **No Documentation**: The function has no docstring, comments, or documentation specifying that zero vectors are not allowed as inputs. Users have no way of knowing this limitation exists.

3. **Poor Error Handling**: The function allows the low-level arithmetic error to surface instead of providing a descriptive error message explaining that cosine similarity is undefined for zero vectors.

4. **Valid Python Input**: Zero vectors (like `[0.0, 0.0]`) are valid Python lists that users might reasonably pass to the function, especially in scenarios like:
   - Initialization of vectors
   - Empty or missing embeddings
   - Result of certain mathematical operations

5. **Internal Usage**: The function is used internally in `embeddings.py` line 263 for distance calculations, making robust error handling even more important.

## Relevant Context

The `llm.cosine_similarity` function is located at `/home/npc/miniconda/lib/python3.13/site-packages/llm/__init__.py:458-462` and has the following implementation:

```python
def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = sum(x * x for x in a) ** 0.5
    magnitude_b = sum(x * x for x in b) ** 0.5
    return dot_product / (magnitude_a * magnitude_b)
```

The function is used internally by the LLM package for computing distance scores in the embeddings module. It's a public function accessible directly via `llm.cosine_similarity()`, though it's not included in the module's `__all__` list.

Testing confirmed that the function works correctly for normal cases:
- Normal vectors: Returns correct similarity values
- Orthogonal vectors: Returns 0.0
- Identical vectors: Returns 1.0
- Opposite vectors: Returns -1.0

The bug only occurs when at least one input vector has zero magnitude.

## Proposed Fix

```diff
--- a/llm/__init__.py
+++ b/llm/__init__.py
@@ -458,6 +458,10 @@ def decode(binary):
 def cosine_similarity(a, b):
     dot_product = sum(x * y for x, y in zip(a, b))
     magnitude_a = sum(x * x for x in a) ** 0.5
     magnitude_b = sum(x * x for x in b) ** 0.5
+    if magnitude_a == 0 or magnitude_b == 0:
+        raise ValueError(
+            "cosine_similarity is undefined for zero vectors"
+        )
     return dot_product / (magnitude_a * magnitude_b)
```