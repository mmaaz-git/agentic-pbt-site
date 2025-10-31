# Bug Report: llm.cosine_similarity Division by Zero on Zero Magnitude Vectors

**Target**: `llm.cosine_similarity`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `cosine_similarity` function crashes with a `ZeroDivisionError` when either input vector has zero magnitude (all elements are zero), instead of handling this mathematically undefined case gracefully.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, assume
import llm

@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=1),
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=1)
)
def test_cosine_similarity_handles_zero_vectors(a, b):
    assume(len(a) == len(b))
    try:
        result = llm.cosine_similarity(a, b)
        if sum(x * x for x in a) > 0 and sum(x * x for x in b) > 0:
            assert -1 <= result <= 1
    except ZeroDivisionError:
        assert False, "Should handle zero vectors gracefully"

if __name__ == "__main__":
    test_cosine_similarity_handles_zero_vectors()
```

<details>

<summary>
**Failing input**: `a=[0.0], b=[0.0]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 14, in test_cosine_similarity_handles_zero_vectors
    result = llm.cosine_similarity(a, b)
  File "/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/llm/__init__.py", line 462, in cosine_similarity
    return dot_product / (magnitude_a * magnitude_b)
           ~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ZeroDivisionError: float division by zero

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 21, in <module>
    test_cosine_similarity_handles_zero_vectors()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 8, in test_cosine_similarity_handles_zero_vectors
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=1),
               ^^^
  File "/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 18, in test_cosine_similarity_handles_zero_vectors
    assert False, "Should handle zero vectors gracefully"
           ^^^^^
AssertionError: Should handle zero vectors gracefully
Falsifying example: test_cosine_similarity_handles_zero_vectors(
    a=[0.0],
    b=[0.0],
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

import llm
cosine_similarity = llm.cosine_similarity

# Test case with zero vector
a = [0.0, 0.0, 0.0]
b = [1.0, 2.0, 3.0]

print(f"Testing cosine_similarity with:")
print(f"a = {a}")
print(f"b = {b}")
print()

try:
    result = cosine_similarity(a, b)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")
```

<details>

<summary>
ZeroDivisionError when computing cosine similarity with zero vector
</summary>
```
Testing cosine_similarity with:
a = [0.0, 0.0, 0.0]
b = [1.0, 2.0, 3.0]

Error occurred: ZeroDivisionError: float division by zero
```
</details>

## Why This Is A Bug

The function crashes with an unhelpful `ZeroDivisionError` instead of gracefully handling a mathematically undefined operation. While cosine similarity is indeed undefined for zero vectors mathematically, a production API should:

1. **Provide meaningful error messages**: The current `ZeroDivisionError` doesn't explain that the issue is zero-magnitude vectors
2. **Handle common edge cases**: Zero vectors commonly occur in real applications:
   - Sparse data representations (e.g., empty document embeddings)
   - Initial values in machine learning pipelines
   - Text embeddings for empty strings
   - Missing or placeholder data

3. **Follow defensive programming principles**: The function accepts lists of numbers as valid input but crashes on certain valid lists without input validation

The function has no documentation whatsoever, making it impossible for users to anticipate this behavior. The lack of graceful error handling suggests this is an oversight rather than intentional design.

## Relevant Context

The function is located at `/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/llm/__init__.py:458-462` with the following implementation:

```python
def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = sum(x * x for x in a) ** 0.5
    magnitude_b = sum(x * x for x in b) ** 0.5
    return dot_product / (magnitude_a * magnitude_b)
```

The mathematical formula for cosine similarity is: cos(θ) = (A · B) / (||A|| × ||B||)

When either ||A|| = 0 or ||B|| = 0, the denominator becomes zero, making the operation undefined. Standard libraries typically handle this by either:
- Returning a special value (0.0, NaN, or None)
- Raising a descriptive ValueError

## Proposed Fix

```diff
--- a/llm/__init__.py
+++ b/llm/__init__.py
@@ -458,6 +458,10 @@ def cosine_similarity(a, b):
     dot_product = sum(x * y for x, y in zip(a, b))
     magnitude_a = sum(x * x for x in a) ** 0.5
     magnitude_b = sum(x * x for x in b) ** 0.5
+
+    if magnitude_a == 0.0 or magnitude_b == 0.0:
+        raise ValueError("Cosine similarity is undefined for zero magnitude vectors")
+
     return dot_product / (magnitude_a * magnitude_b)
```