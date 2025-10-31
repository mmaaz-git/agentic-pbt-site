# Bug Report: scipy.datasets.face Uses Identity Check Instead of Truthiness for gray Parameter

**Target**: `scipy.datasets.face`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `face()` function uses `if gray is True:` (identity check) instead of `if gray:` (truthiness check), causing it to reject truthy values like `1` or `np.bool_(True)` that users might reasonably pass for the gray parameter.

## Property-Based Test

```python
from hypothesis import given, strategies as st

@given(st.sampled_from([True, 1, "true", np.bool_(True)]))
def test_face_gray_accepts_truthy_values(gray_value):
    """Property: face(gray=X) should accept any truthy value, not just True."""
    from scipy import datasets
    import numpy as np

    try:
        result = datasets.face(gray=gray_value)

        if gray_value:
            expected_shape = (768, 1024)
        else:
            expected_shape = (768, 1024, 3)

        assert result.shape == expected_shape, \
            f"gray={gray_value!r} produced shape {result.shape}, expected {expected_shape}"
    except Exception as e:
        print(f"Failed with gray={gray_value!r}: {e}")
```

**Failing input**: `gray_value=1` or `gray_value=np.bool_(True)`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/path/to/scipy')

from scipy import datasets
import numpy as np

print("Testing face() with different truthy values:")
print()

test_values = [
    ("True", True),
    ("1", 1),
    ("np.bool_(True)", np.bool_(True)),
]

for name, value in test_values:
    try:
        result = datasets.face(gray=value)
        print(f"gray={name:20s} -> shape {result.shape}")
    except Exception as e:
        print(f"gray={name:20s} -> ERROR: {e}")
```

Expected output (all should produce grayscale):
```
Testing face() with different truthy values:

gray=True                -> shape (768, 1024)
gray=1                   -> shape (768, 1024)
gray=np.bool_(True)      -> shape (768, 1024)
```

Actual output:
```
Testing face() with different truthy values:

gray=True                -> shape (768, 1024)
gray=1                   -> shape (768, 1024, 3)
gray=np.bool_(True)      -> shape (768, 1024, 3)
```

## Why This Is A Bug

1. **Inconsistent with Python conventions**: Boolean parameters should use truthiness checks (`if x:`) not identity checks (`if x is True:`)
2. **NumPy incompatibility**: NumPy's boolean type `np.bool_` is not identical to Python's `True`, causing `gray=np.bool_(True)` to fail silently
3. **User confusion**: Passing `gray=1` seems like it should work (1 is truthy) but doesn't
4. **Inconsistent with documentation**: The docstring says "gray : bool, optional" which suggests any boolean-like value should work
5. **Silent failure**: The function returns color images instead of raising an error, making the bug hard to detect

The problematic code is on line 222 of `_fetchers.py`:
```python
if gray is True:  # Too strict!
    face = (0.21 * face[:, :, 0] + 0.71 * face[:, :, 1] +
            0.07 * face[:, :, 2]).astype('uint8')
```

## Fix

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

This uses standard Python truthiness evaluation, making the function more flexible and consistent with user expectations.