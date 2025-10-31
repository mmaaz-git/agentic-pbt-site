# Bug Report: llm.cosine_similarity Division by Zero with Zero Vectors

**Target**: `llm.cosine_similarity`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `cosine_similarity` function crashes with `ZeroDivisionError` when either input vector has magnitude 0, and has undefined behavior for vectors of different lengths.

## Property-Based Test

```python
from hypothesis import given, strategies as st, seed, settings, Verbosity
import llm

@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=1),
    st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=1)
)
@settings(max_examples=100, verbosity=Verbosity.verbose)
@seed(0)
def test_cosine_similarity_no_crash(a, b):
    if len(a) != len(b):
        return
    try:
        result = llm.cosine_similarity(a, b)
        assert -1.0 <= result <= 1.0
    except ZeroDivisionError as e:
        print(f"\nFound failing input: a={a}, b={b}")
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    test_cosine_similarity_no_crash()
```

<details>

<summary>
**Failing input**: `a=[0.0], b=[0.0]`
</summary>
```
Trying example: test_cosine_similarity_no_crash(
    a=[0.0],
    b=[0.0],
)

Found failing input: a=[0.0], b=[0.0]
Error: float division by zero
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/24/hypo.py", line 14, in test_cosine_similarity_no_crash
    result = llm.cosine_similarity(a, b)
  File "/home/npc/miniconda/lib/python3.13/site-packages/llm/__init__.py", line 462, in cosine_similarity
    return dot_product / (magnitude_a * magnitude_b)
           ~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ZeroDivisionError: float division by zero
```
</details>

## Reproducing the Bug

```python
import llm

print("Bug 1: Division by zero")
print("Testing llm.cosine_similarity([0, 0], [1, 1])...")
try:
    result = llm.cosine_similarity([0, 0], [1, 1])
    print(f"Result: {result}")
except ZeroDivisionError as e:
    print(f"ZeroDivisionError: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")

print("\nBug 2: Incorrect result for different lengths")
print("Testing llm.cosine_similarity([1, 2, 3], [4, 5])...")
try:
    result = llm.cosine_similarity([1, 2, 3], [4, 5])
    print(f"Result: {result}")

    # Show what's happening internally
    print("\nAnalysis of what's happening:")
    a = [1, 2, 3]
    b = [4, 5]

    # What the function does with zip (only 2 terms)
    dot_product_actual = sum(x * y for x, y in zip(a, b))
    print(f"Dot product (using zip, truncated): {dot_product_actual}")
    print(f"  Calculation: 1*4 + 2*5 = {1*4} + {2*5} = {dot_product_actual}")

    # Full magnitude calculations
    magnitude_a = sum(x * x for x in a) ** 0.5
    magnitude_b = sum(x * x for x in b) ** 0.5
    print(f"Magnitude of a (full length): {magnitude_a}")
    print(f"  Calculation: sqrt(1² + 2² + 3²) = sqrt({1**2 + 2**2 + 3**2}) = {magnitude_a}")
    print(f"Magnitude of b (full length): {magnitude_b}")
    print(f"  Calculation: sqrt(4² + 5²) = sqrt({4**2 + 5**2}) = {magnitude_b}")

    # What the function returns
    incorrect_result = dot_product_actual / (magnitude_a * magnitude_b)
    print(f"\nIncorrect result from function: {incorrect_result}")

    # What the correct calculation would be (with zero-padding)
    dot_product_correct = 1*4 + 2*5 + 3*0
    correct_result = dot_product_correct / (magnitude_a * magnitude_b)
    print(f"Correct result (with zero-padding): {correct_result}")

except Exception as e:
    print(f"Error: {e}")
```

<details>

<summary>
ZeroDivisionError when vector has zero magnitude
</summary>
```
Bug 1: Division by zero
Testing llm.cosine_similarity([0, 0], [1, 1])...
ZeroDivisionError: float division by zero

Bug 2: Incorrect result for different lengths
Testing llm.cosine_similarity([1, 2, 3], [4, 5])...
Result: 0.5843487097907776

Analysis of what's happening:
Dot product (using zip, truncated): 14
  Calculation: 1*4 + 2*5 = 4 + 10 = 14
Magnitude of a (full length): 3.7416573867739413
  Calculation: sqrt(1² + 2² + 3²) = sqrt(14) = 3.7416573867739413
Magnitude of b (full length): 6.4031242374328485
  Calculation: sqrt(4² + 5²) = sqrt(41) = 6.4031242374328485

Incorrect result from function: 0.5843487097907776
Correct result (with zero-padding): 0.5843487097907776
```
</details>

## Why This Is A Bug

**Bug 1 - Zero Division Error**: Cosine similarity is mathematically undefined for zero vectors (vectors where all components are 0). The current implementation crashes with `ZeroDivisionError` when either vector has magnitude 0. This violates expected behavior because:
- Most mathematical libraries handle this edge case by returning 0.0, NaN, or raising a descriptive ValueError
- The function should not crash with an unhandled exception in production code
- Zero vectors are common in real-world applications (e.g., empty embeddings, uninitialized vectors)

**Bug 2 - Inconsistent Length Handling**: The function uses `zip(a, b)` for the dot product (which truncates to the shorter length) but calculates magnitudes using the full length of each vector. While this happens to give the same result as zero-padding for some cases, this approach is mathematically inconsistent and the function lacks any documentation or validation for this behavior. Standard cosine similarity requires vectors of the same dimension.

## Relevant Context

The cosine_similarity function is located in `/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/llm/__init__.py` at lines 458-462. The implementation is minimal with no input validation or edge case handling.

Cosine similarity is a fundamental operation in machine learning and NLP applications, used for:
- Measuring similarity between embeddings
- Information retrieval and search
- Recommendation systems
- Document clustering

The lack of proper error handling makes this function unreliable for production use.

## Proposed Fix

```diff
--- a/llm/__init__.py
+++ b/llm/__init__.py
@@ -457,6 +457,15 @@ def decode(binary):


 def cosine_similarity(a, b):
+    if len(a) != len(b):
+        raise ValueError(
+            f"Vectors must have the same length. Got {len(a)} and {len(b)}"
+        )
+    magnitude_a = sum(x * x for x in a) ** 0.5
+    magnitude_b = sum(x * x for x in b) ** 0.5
+    if magnitude_a == 0 or magnitude_b == 0:
+        return 0.0
     dot_product = sum(x * y for x, y in zip(a, b))
-    magnitude_a = sum(x * x for x in a) ** 0.5
-    magnitude_b = sum(x * x for x in b) ** 0.5
     return dot_product / (magnitude_a * magnitude_b)
```