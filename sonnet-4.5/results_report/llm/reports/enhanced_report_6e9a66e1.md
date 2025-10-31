# Bug Report: llm.cosine_similarity ZeroDivisionError with Zero Vectors

**Target**: `llm.cosine_similarity`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `cosine_similarity` function in the llm package crashes with a `ZeroDivisionError` when either input vector contains all zeros, failing to handle this edge case gracefully despite zero vectors being valid mathematical inputs.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
import llm


@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=100)
)
def test_cosine_similarity_no_crash_with_zero_vectors(a):
    b = [0.0] * len(a)
    result = llm.cosine_similarity(a, b)
    assert isinstance(result, (int, float))

# Run the test
if __name__ == "__main__":
    test_cosine_similarity_no_crash_with_zero_vectors()
```

<details>

<summary>
**Failing input**: `a=[0.0]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 18, in <module>
    test_cosine_similarity_no_crash_with_zero_vectors()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 9, in test_cosine_similarity_no_crash_with_zero_vectors
    st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=100)
               ^^^
  File "/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/57/hypo.py", line 13, in test_cosine_similarity_no_crash_with_zero_vectors
    result = llm.cosine_similarity(a, b)
  File "/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/llm/__init__.py", line 462, in cosine_similarity
    return dot_product / (magnitude_a * magnitude_b)
           ~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ZeroDivisionError: float division by zero
Falsifying example: test_cosine_similarity_no_crash_with_zero_vectors(
    a=[0.0],
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

import llm

# Test case with zero vector
a = [1.0, 2.0, 3.0]
b = [0.0, 0.0, 0.0]

print(f"Testing llm.cosine_similarity with:")
print(f"  a = {a}")
print(f"  b = {b}")
print()

try:
    result = llm.cosine_similarity(a, b)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
```

<details>

<summary>
ZeroDivisionError when b is zero vector
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/57/repo.py", line 16, in <module>
    result = llm.cosine_similarity(a, b)
  File "/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/llm/__init__.py", line 462, in cosine_similarity
    return dot_product / (magnitude_a * magnitude_b)
           ~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ZeroDivisionError: float division by zero
Testing llm.cosine_similarity with:
  a = [1.0, 2.0, 3.0]
  b = [0.0, 0.0, 0.0]

Error: ZeroDivisionError: float division by zero
```
</details>

## Why This Is A Bug

This is a bug because the function fails to handle a mathematically valid edge case. Zero vectors are legitimate vectors that commonly occur in real-world applications such as:

1. **Sparse embeddings** - Where many dimensions may be zero
2. **Initialization states** - Before embeddings are populated
3. **Missing data** - When embeddings are unavailable for certain items
4. **Null representations** - Intentional zero vectors representing absence

The function crashes with an unhelpful `ZeroDivisionError` that doesn't indicate the actual problem (zero magnitude vectors). While cosine similarity is mathematically undefined for zero vectors, a robust implementation should either:
- Return a sentinel value like `0.0` or `float('nan')`
- Raise a descriptive `ValueError` explaining that cosine similarity is undefined for zero vectors

The current implementation violates the principle of least surprise and provides poor developer experience.

## Relevant Context

The `cosine_similarity` function is located at `/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/llm/__init__.py:458-462` and implements the standard cosine similarity formula without any input validation or edge case handling.

The function has no documentation, type hints, or comments explaining expected behavior. This utility function is likely used for comparing embeddings from language models, where zero vectors may occur when:
- Embeddings fail to generate
- Default/placeholder embeddings are used
- Sparse representations have all-zero segments

Other implementations handle this case:
- **NumPy/SciPy**: Various approaches including returning NaN
- **scikit-learn**: Has discussions about appropriate handling (see GitHub issue #15256)
- **PyTorch**: Adds small epsilon to denominators to avoid division by zero

## Proposed Fix

```diff
def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = sum(x * x for x in a) ** 0.5
    magnitude_b = sum(x * x for x in b) ** 0.5
+
+   # Handle zero vectors - cosine similarity undefined
+   if magnitude_a == 0.0 or magnitude_b == 0.0:
+       return 0.0  # Or float('nan') for mathematical correctness
+
    return dot_product / (magnitude_a * magnitude_b)
```