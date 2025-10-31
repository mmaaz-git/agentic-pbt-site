# Bug Report: scipy.fftpack.next_fast_len Accepts Zero Despite Documentation Requiring Positive Integer

**Target**: `scipy.fftpack.next_fast_len`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`scipy.fftpack.next_fast_len(0)` returns 0 instead of raising a `ValueError`, violating its documented API contract that explicitly requires the target parameter to be a positive integer.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
import scipy.fftpack

@given(st.integers(min_value=-1000, max_value=0))
@settings(max_examples=100)
def test_next_fast_len_rejects_non_positive(target):
    """Test that next_fast_len raises ValueError for non-positive integers."""
    try:
        result = scipy.fftpack.next_fast_len(target)
        if target <= 0:
            assert False, f"Should raise ValueError for target={target} but returned {result}"
    except (ValueError, RuntimeError):
        # Expected behavior for non-positive values
        pass

if __name__ == "__main__":
    test_next_fast_len_rejects_non_positive()
```

<details>

<summary>
**Failing input**: `target=0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/20/hypo.py", line 17, in <module>
    test_next_fast_len_rejects_non_positive()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/20/hypo.py", line 5, in test_next_fast_len_rejects_non_positive
    @settings(max_examples=100)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/20/hypo.py", line 11, in test_next_fast_len_rejects_non_positive
    assert False, f"Should raise ValueError for target={target} but returned {result}"
           ^^^^^
AssertionError: Should raise ValueError for target=0 but returned 0
Falsifying example: test_next_fast_len_rejects_non_positive(
    target=0,
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/20/hypo.py:10
```
</details>

## Reproducing the Bug

```python
import scipy.fftpack
import numpy as np

# Test with 0 - should raise ValueError according to documentation
print("Testing scipy.fftpack.next_fast_len(0):")
result = scipy.fftpack.next_fast_len(0)
print(f"next_fast_len(0) = {result}")
print(f"Type of result: {type(result)}")
print()

# Test with -1 - should raise ValueError
print("Testing scipy.fftpack.next_fast_len(-1):")
try:
    scipy.fftpack.next_fast_len(-1)
    print("next_fast_len(-1) did not raise an error")
except ValueError as e:
    print(f"next_fast_len(-1) raises ValueError: {e}")
print()

# Test using the result in fft
print("Testing using result=0 in scipy.fftpack.fft:")
try:
    x = np.array([1., 2., 3.])
    fft_result = scipy.fftpack.fft(x, n=result)
    print(f"fft with n={result} succeeded")
except ValueError as e:
    print(f"fft with n={result} raises ValueError: {e}")
print()

# Additional tests to show expected behavior
print("Testing with positive values:")
for val in [1, 2, 5, 10, 100]:
    fast_len = scipy.fftpack.next_fast_len(val)
    print(f"next_fast_len({val}) = {fast_len}")
```

<details>

<summary>
Output demonstrating inconsistent validation and downstream error
</summary>
```
Testing scipy.fftpack.next_fast_len(0):
next_fast_len(0) = 0
Type of result: <class 'int'>

Testing scipy.fftpack.next_fast_len(-1):
next_fast_len(-1) raises ValueError: Target length must be positive

Testing using result=0 in scipy.fftpack.fft:
fft with n=0 raises ValueError: invalid number of data points (0) specified

Testing with positive values:
next_fast_len(1) = 1
next_fast_len(2) = 2
next_fast_len(5) = 5
next_fast_len(10) = 10
next_fast_len(100) = 100
```
</details>

## Why This Is A Bug

This violates expected behavior in multiple critical ways:

1. **Documentation Contract Violation**: The function's docstring explicitly states "target : int - Length to start searching from. Must be a positive integer." In mathematics and computer science, positive integers are defined as the set {1, 2, 3, ...}, which explicitly excludes 0. Zero is neither positive nor negative.

2. **Inconsistent Input Validation**: The function correctly raises `ValueError: Target length must be positive` for negative inputs (e.g., -1), but silently accepts 0 and returns 0. This inconsistency suggests incomplete input validation rather than intentional design.

3. **Invalid Return Value**: The function is documented to return "The first 5-smooth number greater than or equal to `target`." A 5-smooth number (also called a Hamming number) must be a positive integer whose only prime factors are 2, 3, and 5. Zero is not a 5-smooth number by definition, making the returned value mathematically incorrect.

4. **Unusable Result**: When the returned value (0) is used in downstream FFT operations like `scipy.fftpack.fft(x, n=0)`, it causes a `ValueError: invalid number of data points (0) specified`. This demonstrates that 0 is not a valid FFT length, confirming it should never be returned.

5. **Purpose Violation**: The function's stated purpose is to "Find the next fast size of input data to `fft`, for zero-padding, etc." Zero-padding to size 0 is nonsensical and defeats the optimization purpose of the function.

## Relevant Context

The `scipy.fftpack.next_fast_len` function is designed to find optimal FFT sizes by returning 5-smooth numbers (composite of prime factors 2, 3, and 5). These sizes allow for more efficient FFT computation compared to arbitrary sizes, especially prime numbers.

The function implementation (from scipy/fftpack/_helper.py) calls an internal `_helper.good_size(target, True)` function. This underlying function correctly validates negative inputs but fails to validate zero, creating the inconsistency.

**Documentation References**:
- [SciPy FFTpack documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.next_fast_len.html)
- The function was added in SciPy version 0.18.0 as noted in the docstring

**Mathematical Context**:
- Positive integers: {1, 2, 3, 4, ...}
- 5-smooth numbers (Hamming numbers): {1, 2, 3, 4, 5, 6, 8, 9, 10, 12, ...}
- Zero is neither a positive integer nor a 5-smooth number

## Proposed Fix

```diff
--- a/scipy/fftpack/_helper.py
+++ b/scipy/fftpack/_helper.py
@@ -99,6 +99,8 @@ def next_fast_len(target):
     >>> b = fftpack.fft(a, 16384)

     """
+    if target <= 0:
+        raise ValueError("Target length must be positive")
     # Real transforms use regular sizes so this is backwards compatible
     return _helper.good_size(target, True)
```