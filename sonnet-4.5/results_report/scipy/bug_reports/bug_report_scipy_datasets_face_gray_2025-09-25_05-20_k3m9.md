# Bug Report: scipy.datasets.face() Silently Ignores Truthy Non-Boolean Values

**Target**: `scipy.datasets.face`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `face()` function uses identity comparison (`is True`) instead of truthiness checking for the `gray` parameter, causing it to silently ignore truthy non-boolean values like `1` and return a color image instead of grayscale, violating user expectations without any error message.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np

def face_logic(face_array, gray):
    if gray is True:
        return (0.21 * face_array[:, :, 0] + 0.71 * face_array[:, :, 1] +
                0.07 * face_array[:, :, 2]).astype('uint8')
    return face_array

@given(st.integers(min_value=1, max_value=10))
def test_face_gray_truthy_values(val):
    mock_face = np.random.randint(0, 256, size=(768, 1024, 3), dtype='uint8')
    result = face_logic(mock_face, gray=val)

    if val:
        assert result.ndim == 2, \
            f"Truthy value {val} should trigger grayscale conversion but returned shape {result.shape}"
```

**Failing input**: `gray=1` (or any truthy non-boolean value)

## Reproducing the Bug

```python
import numpy as np

def face_logic(face_array, gray):
    if gray is True:
        face = (0.21 * face_array[:, :, 0] + 0.71 * face_array[:, :, 1] +
                0.07 * face_array[:, :, 2]).astype('uint8')
        return face
    return face_array

mock_face = np.random.randint(0, 256, size=(768, 1024, 3), dtype='uint8')

result = face_logic(mock_face, gray=1)
print(f"face(gray=1) shape: {result.shape}")
print(f"Expected: (768, 1024) for grayscale")
print(f"Actual: {result.shape} (color image)")
```

## Why This Is A Bug

The function signature documents `gray` as "bool, optional" but uses identity comparison (`is True`) instead of truthiness checking. This creates two problems:

1. **Silently accepts invalid input**: When users pass truthy non-boolean values like `gray=1` (common when interfacing with C code or numpy), the function accepts the input without error but produces unexpected results.

2. **Violates Python conventions**: Python idiomatically uses truthiness checking (`if gray:`) rather than identity comparison (`if gray is True:`). The current implementation is inconsistent with user expectations.

The function should either:
- Use truthiness checking to accept truthy values (Pythonic approach), OR
- Validate that `gray` is actually a boolean and raise `TypeError` for invalid inputs

Currently it does neither, leading to silent failures.

## Fix

```diff
diff --git a/scipy/datasets/_fetchers.py b/scipy/datasets/_fetchers.py
index 1234567..abcdefg 100644
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

Alternatively, if strict type checking is desired:

```diff
diff --git a/scipy/datasets/_fetchers.py b/scipy/datasets/_fetchers.py
index 1234567..abcdefg 100644
--- a/scipy/datasets/_fetchers.py
+++ b/scipy/datasets/_fetchers.py
@@ -214,6 +214,9 @@ def face(gray=False):

     """
     import bz2
+    if not isinstance(gray, bool):
+        raise TypeError(f"gray parameter must be bool, got {type(gray).__name__}")
+
     fname = fetch_data("face.dat")
     with open(fname, 'rb') as f:
         rawdata = f.read()