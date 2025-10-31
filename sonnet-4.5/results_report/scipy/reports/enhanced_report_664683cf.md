# Bug Report: scipy.datasets.face() Silently Ignores Truthy Non-Boolean Values for Gray Parameter

**Target**: `scipy.datasets.face`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `scipy.datasets.face()` function uses identity comparison (`gray is True`) instead of truthiness checking, causing it to silently return color images when passed truthy non-boolean values like `1`, `"yes"`, or `[1,2,3]`, despite the user's clear intent to get grayscale output.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Property-based test demonstrating scipy.datasets.face() bug with truthy values."""

from hypothesis import given, strategies as st, settings
import scipy.datasets

@given(st.one_of(
    st.integers(min_value=1, max_value=10),
    st.text(min_size=1, max_size=5).filter(bool),
    st.lists(st.integers(), min_size=1, max_size=3)
))
@settings(max_examples=10)
def test_face_gray_truthy_values(val):
    """Test that truthy values for gray parameter should produce grayscale images."""
    result = scipy.datasets.face(gray=val)

    # All truthy values should trigger grayscale conversion
    if val:
        assert result.ndim == 2, \
            f"Truthy value {val!r} (type: {type(val).__name__}) should trigger grayscale conversion but returned shape {result.shape}"
        assert result.shape == (768, 1024), \
            f"Expected grayscale shape (768, 1024), got {result.shape}"
    else:
        # Falsy values should return color
        assert result.ndim == 3, \
            f"Falsy value {val!r} should return color image but returned shape {result.shape}"
        assert result.shape == (768, 1024, 3), \
            f"Expected color shape (768, 1024, 3), got {result.shape}"

if __name__ == "__main__":
    test_face_gray_truthy_values()
```

<details>

<summary>
**Failing input**: `val=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 31, in <module>
    test_face_gray_truthy_values()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 8, in test_face_gray_truthy_values
    st.integers(min_value=1, max_value=10),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/19/hypo.py", line 19, in test_face_gray_truthy_values
    assert result.ndim == 2, \
           ^^^^^^^^^^^^^^^^
AssertionError: Truthy value 1 (type: int) should trigger grayscale conversion but returned shape (768, 1024, 3)
Falsifying example: test_face_gray_truthy_values(
    val=1,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction of scipy.datasets.face() bug with truthy non-boolean values."""

import scipy.datasets
import numpy as np

print("Testing scipy.datasets.face() with various truthy/falsy values:\n")
print("="*60)

# Test with boolean True (expected to work)
face_true = scipy.datasets.face(gray=True)
print(f"face(gray=True) shape: {face_true.shape}")
print(f"  Expected: (768, 1024) for grayscale")
print(f"  Result: {'✓ CORRECT' if face_true.ndim == 2 else '✗ INCORRECT'}\n")

# Test with boolean False (expected to return color)
face_false = scipy.datasets.face(gray=False)
print(f"face(gray=False) shape: {face_false.shape}")
print(f"  Expected: (768, 1024, 3) for color")
print(f"  Result: {'✓ CORRECT' if face_false.ndim == 3 else '✗ INCORRECT'}\n")

# Test with integer 1 (truthy, should trigger grayscale but doesn't)
face_one = scipy.datasets.face(gray=1)
print(f"face(gray=1) shape: {face_one.shape}")
print(f"  Expected: (768, 1024) for grayscale (truthy value)")
print(f"  Actual: {face_one.shape}")
print(f"  Result: {'✓ CORRECT' if face_one.ndim == 2 else '✗ BUG - Returns color despite truthy value'}\n")

# Test with integer 0 (falsy, should return color)
face_zero = scipy.datasets.face(gray=0)
print(f"face(gray=0) shape: {face_zero.shape}")
print(f"  Expected: (768, 1024, 3) for color (falsy value)")
print(f"  Result: {'✓ CORRECT' if face_zero.ndim == 3 else '✗ INCORRECT'}\n")

# Test with string "yes" (truthy, should trigger grayscale but doesn't)
face_string = scipy.datasets.face(gray="yes")
print(f"face(gray='yes') shape: {face_string.shape}")
print(f"  Expected: (768, 1024) for grayscale (truthy value)")
print(f"  Actual: {face_string.shape}")
print(f"  Result: {'✓ CORRECT' if face_string.ndim == 2 else '✗ BUG - Returns color despite truthy value'}\n")

# Test with list [1,2,3] (truthy, should trigger grayscale but doesn't)
face_list = scipy.datasets.face(gray=[1,2,3])
print(f"face(gray=[1,2,3]) shape: {face_list.shape}")
print(f"  Expected: (768, 1024) for grayscale (truthy value)")
print(f"  Actual: {face_list.shape}")
print(f"  Result: {'✓ CORRECT' if face_list.ndim == 2 else '✗ BUG - Returns color despite truthy value'}\n")

print("="*60)
print("\nROOT CAUSE:")
print("The function uses 'if gray is True:' instead of 'if gray:'")
print("This causes it to only accept the exact boolean True,")
print("ignoring all other truthy values without any error message.")
```

<details>

<summary>
Demonstration showing truthy values silently fail to trigger grayscale conversion
</summary>
```
Testing scipy.datasets.face() with various truthy/falsy values:

============================================================
face(gray=True) shape: (768, 1024)
  Expected: (768, 1024) for grayscale
  Result: ✓ CORRECT

face(gray=False) shape: (768, 1024, 3)
  Expected: (768, 1024, 3) for color
  Result: ✓ CORRECT

face(gray=1) shape: (768, 1024, 3)
  Expected: (768, 1024) for grayscale (truthy value)
  Actual: (768, 1024, 3)
  Result: ✗ BUG - Returns color despite truthy value

face(gray=0) shape: (768, 1024, 3)
  Expected: (768, 1024, 3) for color (falsy value)
  Result: ✓ CORRECT

face(gray='yes') shape: (768, 1024, 3)
  Expected: (768, 1024) for grayscale (truthy value)
  Actual: (768, 1024, 3)
  Result: ✗ BUG - Returns color despite truthy value

face(gray=[1,2,3]) shape: (768, 1024, 3)
  Expected: (768, 1024) for grayscale (truthy value)
  Actual: (768, 1024, 3)
  Result: ✗ BUG - Returns color despite truthy value

============================================================

ROOT CAUSE:
The function uses 'if gray is True:' instead of 'if gray:'
This causes it to only accept the exact boolean True,
ignoring all other truthy values without any error message.
```
</details>

## Why This Is A Bug

This violates expected Python behavior in three critical ways:

1. **Type Contract Violation Without Validation**: The docstring explicitly declares `gray : bool, optional`, establishing a type contract. However, the function accepts ANY type without validation or error. When users pass non-boolean values (e.g., `gray=1` from C interfaces, NumPy operations, or configuration files), the function silently accepts them but produces unexpected results.

2. **Identity Check vs Truthiness**: The implementation uses `if gray is True:` (identity comparison) instead of the Pythonic `if gray:` (truthiness check). This is highly unusual in Python where truthiness is the standard convention. Users familiar with Python expect truthy values to be treated as True, especially when no type validation occurs.

3. **Silent Failure Pattern**: The function neither validates input types nor handles truthy values correctly. It silently returns the wrong image format without any warning or error. This makes bugs hard to detect - users may process thousands of images thinking they're getting grayscale when they're actually getting color.

The actual implementation at `/home/npc/.local/lib/python3.13/site-packages/scipy/datasets/_fetchers.py:222` shows:
```python
if gray is True:
    face = (0.21 * face[:, :, 0] + 0.71 * face[:, :, 1] +
            0.07 * face[:, :, 2]).astype('uint8')
```

## Relevant Context

- **Source location**: `/home/npc/.local/lib/python3.13/site-packages/scipy/datasets/_fetchers.py:183-225`
- **Function signature**: `def face(gray=False)`
- **Documentation**: Parameter documented as `gray : bool, optional`
- **Common use cases where this bug manifests**:
  - Interfacing with C extensions that use integers for booleans
  - NumPy operations that produce integer results used as flags
  - Configuration files or command-line arguments parsed as integers
  - Dynamic parameter passing from other systems

This is particularly problematic because `scipy.datasets.face()` is commonly used in tutorials, documentation examples, and test suites where incorrect behavior could propagate unnoticed.

## Proposed Fix

```diff
--- a/scipy/datasets/_fetchers.py
+++ b/scipy/datasets/_fetchers.py
@@ -219,7 +219,7 @@ def face(gray=False):
     face_data = bz2.decompress(rawdata)
     face = frombuffer(face_data, dtype='uint8')
     face.shape = (768, 1024, 3)
-    if gray is True:
+    if gray:
         face = (0.21 * face[:, :, 0] + 0.71 * face[:, :, 1] +
                 0.07 * face[:, :, 2]).astype('uint8')
     return face
```