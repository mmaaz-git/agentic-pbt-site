# Bug Report: scipy.ndimage.binary_closing Violates Extensiveness Property

**Target**: `scipy.ndimage.binary_closing`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`scipy.ndimage.binary_closing` violates the fundamental extensiveness property of morphological closing: X ⊆ closing(X). With the default `border_value=0`, closing can remove True pixels at the image border, contradicting both the mathematical definition and documented behavior that closing "fills holes."

## Property-Based Test

```python
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from hypothesis import strategies as st
import numpy as np
import scipy.ndimage as ndi

@given(
    binary_image=arrays(
        dtype=bool,
        shape=st.tuples(
            st.integers(min_value=5, max_value=20),
            st.integers(min_value=5, max_value=20)
        )
    )
)
@settings(max_examples=200)
def test_binary_closing_superset(binary_image):
    """
    Closing adds points: X ⊆ closing(X)
    Closing can only add points, never remove them.
    """
    closed = ndi.binary_closing(binary_image)
    assert np.all(binary_image <= closed), \
        "Input should be a subset of binary closing result"
```

**Failing input**: `binary_image=np.ones((5, 5), dtype=bool)` (array of all True)

## Reproducing the Bug

```python
import numpy as np
import scipy.ndimage as ndi

input_array = np.ones((5, 5), dtype=bool)

closed = ndi.binary_closing(input_array)

print("Input (all True):")
print(input_array.astype(int))

print("\nClosed:")
print(closed.astype(int))

print(f"\nExtensiveness check: {np.all(input_array <= closed)}")
print(f"Closing REMOVED {np.sum(input_array) - np.sum(closed)} pixels")
```

**Output:**
```
Input (all True):
[[1 1 1 1 1]
 [1 1 1 1 1]
 [1 1 1 1 1]
 [1 1 1 1 1]
 [1 1 1 1 1]]

Closed:
[[0 0 0 0 0]
 [0 1 1 1 0]
 [0 1 1 1 0]
 [0 1 1 1 0]
 [0 0 0 0 0]]

Extensiveness check: False
Closing REMOVED 16 pixels
```

## Why This Is A Bug

This violates three fundamental expectations:

1. **Mathematical correctness**: Morphological closing is defined to be *extensive*, meaning closing(X) ⊇ X. This property should hold regardless of border handling.

2. **Documentation contract**: The documentation states closing "fills holes smaller than the structuring element" - it describes an additive operation, not one that removes pixels.

3. **API consistency**: Related operations like `binary_opening` correctly satisfy their constraint (opening(X) ⊆ X works even at borders with border_value=0).

The root cause is that `binary_closing` uses `border_value=0` by default, inherited from its component operations (`binary_dilation` then `binary_erosion`). This causes the erosion step to remove border pixels that were part of the original image.

## Fix

The fix should ensure closing preserves the extensiveness property. Two approaches:

**Option 1: Change default border_value for binary_closing**
```diff
--- a/scipy/ndimage/_morphology.py
+++ b/scipy/ndimage/_morphology.py
@@ -binary_closing
-def binary_closing(input, structure=None, iterations=1, output=None, origin=0, mask=None, border_value=0, brute_force=False, *, axes=None):
+def binary_closing(input, structure=None, iterations=1, output=None, origin=0, mask=None, border_value=1, brute_force=False, *, axes=None):
```

**Option 2: Document the violation clearly**

If changing the default is deemed too breaking, the documentation should explicitly warn:
```
Notes
-----
Warning: With the default ``border_value=0``, closing may remove True pixels
at the image border, violating the extensiveness property (X ⊆ closing(X)).
Use ``border_value=1`` to preserve this mathematical property.
```

**Recommendation**: Option 1 is preferable as it makes the function mathematically correct by default. Users who want the current behavior can explicitly pass `border_value=0`.