# Bug Report: scipy.ndimage.binary_closing Violates Extensiveness Property of Morphological Closing

**Target**: `scipy.ndimage.binary_closing`
**Severity**: High
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`scipy.ndimage.binary_closing` violates the mathematical extensiveness property (X ⊆ closing(X)) by removing pixels at image borders when using the default `border_value=0`, contradicting both mathematical correctness and the documented behavior that closing "fills holes."

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

if __name__ == "__main__":
    # Run the test
    test_binary_closing_superset()
```

<details>

<summary>
**Failing input**: `binary_image=array([[True, True, True, True, True], [True, True, True, True, True], [True, True, True, True, True], [True, True, True, True, True], [True, True, True, True, True]])`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/5/hypo.py", line 28, in <module>
    test_binary_closing_superset()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/5/hypo.py", line 8, in test_binary_closing_superset
    binary_image=arrays(
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/5/hypo.py", line 23, in test_binary_closing_superset
    assert np.all(binary_image <= closed), \
           ~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Input should be a subset of binary closing result
Falsifying example: test_binary_closing_superset(
    binary_image=array([[ True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True],
           [ True,  True,  True,  True,  True]]),
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
import scipy.ndimage as ndi

# Create an input array of all True values
input_array = np.ones((5, 5), dtype=bool)

# Apply binary_closing with default parameters
closed = ndi.binary_closing(input_array)

print("Input (all True):")
print(input_array.astype(int))

print("\nClosed result:")
print(closed.astype(int))

# Check if extensiveness property holds (input should be subset of closed)
extensiveness_check = np.all(input_array <= closed)
print(f"\nExtensiveness property (X ⊆ closing(X)): {extensiveness_check}")

# Count how many pixels were removed (should be 0 for proper closing)
pixels_removed = np.sum(input_array) - np.sum(closed)
print(f"Pixels removed by closing: {pixels_removed}")

if pixels_removed > 0:
    print(f"\nBUG: Closing REMOVED {pixels_removed} pixels instead of only adding or keeping them!")
```

<details>

<summary>
Bug: binary_closing removes 16 pixels from an all-True 5x5 array
</summary>
```
Input (all True):
[[1 1 1 1 1]
 [1 1 1 1 1]
 [1 1 1 1 1]
 [1 1 1 1 1]
 [1 1 1 1 1]]

Closed result:
[[0 0 0 0 0]
 [0 1 1 1 0]
 [0 1 1 1 0]
 [0 1 1 1 0]
 [0 0 0 0 0]]

Extensiveness property (X ⊆ closing(X)): False
Pixels removed by closing: 16

BUG: Closing REMOVED 16 pixels instead of only adding or keeping them!
```
</details>

## Why This Is A Bug

This behavior violates three critical expectations:

1. **Mathematical Correctness**: Morphological closing is mathematically defined to be *extensive*, meaning closing(X) ⊇ X must always hold. This is a fundamental property in mathematical morphology that distinguishes closing from other operations. The current implementation violates this axiom when True pixels exist at image borders.

2. **Documentation Contract Violation**: The function's docstring explicitly states: "Closing therefore fills holes smaller than the structuring element." This describes a purely additive operation that should never remove existing foreground pixels. The documentation at line 742-743 in `_morphology.py` makes no mention that pixels can be removed.

3. **Inconsistency with Related Operations**: The companion operation `binary_opening` correctly preserves its mathematical constraint (opening(X) ⊆ X) even with `border_value=0`. Users reasonably expect dual operations to handle boundaries consistently.

The root cause is that `binary_closing` performs dilation followed by erosion, both using `border_value=0` by default. During erosion, pixels at the border are compared against the assumed 0-valued pixels outside the image boundary, causing them to be erroneously removed.

## Relevant Context

The implementation is found at `/home/npc/.local/lib/python3.13/site-packages/scipy/ndimage/_morphology.py:823-826`:

```python
def binary_closing(input, structure=None, iterations=1, output=None,
                   origin=0, mask=None, border_value=0, brute_force=False, *,
                   axes=None):
    # ... (documentation and setup)
    tmp = binary_dilation(input, structure, iterations, mask, None,
                          border_value, origin, brute_force, axes=axes)
    return binary_erosion(tmp, structure, iterations, mask, output,
                          border_value, origin, brute_force, axes=axes)
```

Testing with `border_value=1` produces the mathematically correct result where the extensiveness property holds.

Documentation references:
- Mathematical morphology: https://en.wikipedia.org/wiki/Mathematical_morphology
- Closing operation: https://en.wikipedia.org/wiki/Closing_(morphology)

## Proposed Fix

Change the default `border_value` parameter from 0 to 1 for `binary_closing` to ensure mathematical correctness by default:

```diff
--- a/scipy/ndimage/_morphology.py
+++ b/scipy/ndimage/_morphology.py
@@ -676,7 +676,7 @@ def binary_fill_holes(input, structure=None, output=None, origin=0, *, axes=Non

 def binary_closing(input, structure=None, iterations=1, output=None,
-                   origin=0, mask=None, border_value=0, brute_force=False, *,
+                   origin=0, mask=None, border_value=1, brute_force=False, *,
                    axes=None):
     """
     Multidimensional binary closing with the given structuring element.
```

This ensures the extensiveness property X ⊆ closing(X) holds by default. Users who require the current behavior can explicitly pass `border_value=0`.