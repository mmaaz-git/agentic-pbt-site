# Bug Report: llm.cosine_similarity Crashes with Unhelpful Error on Zero-Magnitude Vectors

**Target**: `llm.cosine_similarity`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `cosine_similarity` function crashes with an unhelpful `ZeroDivisionError` when either input vector has zero magnitude, failing to provide users with clear guidance about the mathematical impossibility of computing cosine similarity for zero vectors.

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

if __name__ == "__main__":
    test_cosine_similarity_handles_zero_vectors()
```

<details>

<summary>
**Failing input**: `b=[0.0]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 10, in test_cosine_similarity_handles_zero_vectors
    result = llm.cosine_similarity(a, b)
  File "/home/npc/miniconda/lib/python3.13/site-packages/llm/__init__.py", line 462, in cosine_similarity
    return dot_product / (magnitude_a * magnitude_b)
           ~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ZeroDivisionError: float division by zero

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 18, in <module>
    test_cosine_similarity_handles_zero_vectors()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 6, in test_cosine_similarity_handles_zero_vectors
    def test_cosine_similarity_handles_zero_vectors(b):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 13, in test_cosine_similarity_handles_zero_vectors
    assert False, "Should raise ValueError, not ZeroDivisionError"
           ^^^^^
AssertionError: Should raise ValueError, not ZeroDivisionError
Falsifying example: test_cosine_similarity_handles_zero_vectors(
    b=[0.0],
)
```
</details>

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

<details>

<summary>
Crashes with: ZeroDivisionError: float division by zero
</summary>
```
ZeroDivisionError: float division by zero
```
</details>

## Why This Is A Bug

This violates expected behavior because the function provides no clear error handling for a well-known mathematical edge case. Cosine similarity is mathematically undefined when either vector has zero magnitude (||A|| = 0 or ||B|| = 0), as the formula requires dividing by the product of magnitudes. The current implementation directly performs the division without checking for zero magnitudes, resulting in a generic `ZeroDivisionError` that doesn't explain the actual problem to users.

The function lacks any documentation specifying its behavior with zero vectors, and the raw arithmetic error provides no context about why the computation failed. Users encountering this error must trace through the stack trace and understand the mathematical formula to diagnose that their input contained a zero-magnitude vector.

## Relevant Context

The `cosine_similarity` function is located in `/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/llm/__init__.py` at lines 458-462. It's used internally by the LLM package for computing similarities between embeddings, particularly in the `llm similar` command and `collection.similar()` method.

Zero-magnitude vectors commonly occur in real-world scenarios:
- Empty documents producing zero embeddings
- Failed embedding generation returning default zero vectors
- Initialization values before proper embedding computation
- Masked or filtered vectors where all components are zeroed

Industry standard libraries like scikit-learn handle this case gracefully, either returning 0 or NaN with appropriate documentation. The mathematical formula for cosine similarity is: cos(θ) = (A·B) / (||A|| × ||B||), which is undefined when the denominator is zero.

## Proposed Fix

```diff
def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = sum(x * x for x in a) ** 0.5
    magnitude_b = sum(x * x for x in b) ** 0.5
+
+   if magnitude_a == 0 or magnitude_b == 0:
+       raise ValueError(
+           f"Cannot compute cosine similarity with zero-magnitude vector(s). "
+           f"magnitude(a)={magnitude_a}, magnitude(b)={magnitude_b}"
+       )
+
    return dot_product / (magnitude_a * magnitude_b)
```