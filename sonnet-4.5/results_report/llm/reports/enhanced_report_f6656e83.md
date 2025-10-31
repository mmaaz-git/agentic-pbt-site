# Bug Report: llm.cosine_similarity Silently Accepts Mismatched Vector Lengths

**Target**: `llm.cosine_similarity`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `cosine_similarity` function silently truncates longer vectors when given inputs of different lengths, returning mathematically incorrect results without any warning or error.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st, assume
import llm

@settings(max_examples=500)
@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=1, max_size=10),
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=1, max_size=10)
)
def test_cosine_similarity_length_check(a, b):
    # Skip zero vectors as they cause a different bug (division by zero)
    assume(any(x != 0 for x in a) and any(x != 0 for x in b))

    if len(a) != len(b):
        try:
            result = llm.cosine_similarity(a, b)
            assert False, (
                f"cosine_similarity should reject mismatched lengths but returned {result} "
                f"for a (len={len(a)}) and b (len={len(b)})"
            )
        except ValueError:
            # This is what we expect - the function should raise ValueError for mismatched lengths
            pass

if __name__ == "__main__":
    test_cosine_similarity_length_check()
```

<details>

<summary>
**Failing input**: `a=[0.0, 1.0], b=[1.0]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 25, in <module>
    test_cosine_similarity_length_check()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 5, in test_cosine_similarity_length_check
    @given(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 16, in test_cosine_similarity_length_check
    assert False, (
           ^^^^^
AssertionError: cosine_similarity should reject mismatched lengths but returned 0.0 for a (len=2) and b (len=1)
Falsifying example: test_cosine_similarity_length_check(
    a=[0.0, 1.0],
    b=[1.0],
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/17/hypo.py:14
```
</details>

## Reproducing the Bug

```python
import llm

# Test case from the bug report - mismatched vector lengths
a = [1, 0]
b = [1]
result = llm.cosine_similarity(a, b)

print(f"cosine_similarity({a}, {b}) = {result}")
print(f"This should raise an error for mismatched vector lengths")
print(f"Instead it silently truncates and returns an incorrect result")

# Additional test case - zero vectors causing division by zero
print("\nAdditional test case - zero vectors:")
try:
    c = [0.0]
    d = [0.0, 0.0]
    result2 = llm.cosine_similarity(c, d)
    print(f"cosine_similarity({c}, {d}) = {result2}")
except ZeroDivisionError as e:
    print(f"ZeroDivisionError: {e}")
    print(f"Function crashes on zero vectors")
```

<details>

<summary>
Output demonstrating silent truncation and division by zero bugs
</summary>
```
cosine_similarity([1, 0], [1]) = 1.0
This should raise an error for mismatched vector lengths
Instead it silently truncates and returns an incorrect result

Additional test case - zero vectors:
ZeroDivisionError: float division by zero
Function crashes on zero vectors
```
</details>

## Why This Is A Bug

This violates the mathematical definition of cosine similarity, which is only defined for vectors of equal dimensionality. The function's implementation at `/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/llm/__init__.py:458-462` uses Python's `zip()` function which silently truncates to the shorter vector length:

1. **Mathematical Incorrectness**: Cosine similarity measures the cosine of the angle between two vectors in n-dimensional space. Vectors of different dimensions exist in different spaces and cannot have an angle between them.

2. **Silent Data Corruption**: When given `a=[1, 0]` and `b=[1]`, the function:
   - Uses `zip(a, b)` which yields only `[(1, 1)]`, ignoring the second element of `a`
   - Calculates `dot_product = 1*1 = 1` (incorrect - should use all elements)
   - Calculates `magnitude_a = sqrt(1² + 0²) = 1.0` (uses full vector)
   - Calculates `magnitude_b = sqrt(1²) = 1.0`
   - Returns `1.0` suggesting perfect alignment, when the calculation is mathematically undefined

3. **Production Impact**: The function is actively used in `llm/embeddings.py:263` for computing embedding similarity scores, meaning this bug could cause incorrect search results in embedding-based applications.

4. **Additional Bug**: The function also crashes with `ZeroDivisionError` when given zero vectors, which should instead return a defined value (typically 0 or raise a more descriptive error).

## Relevant Context

- The function is located at `/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/llm/__init__.py:458-462`
- It's used in production code at `llm/embeddings.py:263` in the `distance_score` function for embedding similarity calculations
- Standard libraries like NumPy and SciPy correctly reject mismatched dimensions with clear error messages
- The function lacks any documentation, type hints, or input validation
- Mathematical reference: [Cosine Similarity on Wikipedia](https://en.wikipedia.org/wiki/Cosine_similarity)

## Proposed Fix

```diff
--- a/llm/__init__.py
+++ b/llm/__init__.py
@@ -457,6 +457,13 @@ def decode(binary):


 def cosine_similarity(a, b):
+    if len(a) != len(b):
+        raise ValueError(
+            f"Vectors must have the same length: len(a)={len(a)}, len(b)={len(b)}"
+        )
+    if all(x == 0 for x in a) or all(x == 0 for x in b):
+        # Handle zero vectors gracefully
+        return 0.0
     dot_product = sum(x * y for x, y in zip(a, b))
     magnitude_a = sum(x * x for x in a) ** 0.5
     magnitude_b = sum(x * x for x in b) ** 0.5
```