# Bug Report: scipy.spatial.distance.jensenshannon Division by Zero with All-Zero Vectors

**Target**: `scipy.spatial.distance.jensenshannon`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `jensenshannon` function returns NaN when both input probability vectors contain all zeros, due to division by zero during normalization, violating the fundamental identity property that distance(p, p) should equal 0.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st, settings
from scipy.spatial.distance import jensenshannon


@given(
    st.lists(st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=1, max_size=100)
)
@settings(max_examples=500)
def test_jensenshannon_identity(p_list):
    p = np.array(p_list)
    d = jensenshannon(p, p)
    assert np.isclose(d, 0.0), f"jensenshannon(p, p) should be 0, got {d}"

if __name__ == "__main__":
    test_jensenshannon_identity()
```

<details>

<summary>
**Failing input**: `p_list=[0.0]`
</summary>
```
/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/distance.py:1378: RuntimeWarning: invalid value encountered in divide
  p = p / np.sum(p, axis=axis, keepdims=True)
/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/distance.py:1379: RuntimeWarning: invalid value encountered in divide
  q = q / np.sum(q, axis=axis, keepdims=True)
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 16, in <module>
    test_jensenshannon_identity()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 7, in test_jensenshannon_identity
    st.lists(st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=1, max_size=100)
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 13, in test_jensenshannon_identity
    assert np.isclose(d, 0.0), f"jensenshannon(p, p) should be 0, got {d}"
           ~~~~~~~~~~^^^^^^^^
AssertionError: jensenshannon(p, p) should be 0, got nan
Falsifying example: test_jensenshannon_identity(
    p_list=[0.0],
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from scipy.spatial.distance import jensenshannon

# Test case that crashes: all-zero vectors
p = np.array([0.0, 0.0, 0.0])
q = np.array([0.0, 0.0, 0.0])

print("Testing jensenshannon with all-zero vectors:")
print(f"p = {p}")
print(f"q = {q}")

result = jensenshannon(p, q)
print(f"jensenshannon(p, q) = {result}")
print(f"Result is NaN: {np.isnan(result)}")

# For comparison, test with valid probability vectors
print("\n" + "="*50)
print("Testing jensenshannon with valid probability vectors:")
p_valid = np.array([0.3, 0.3, 0.4])
q_valid = np.array([0.3, 0.3, 0.4])
print(f"p_valid = {p_valid}")
print(f"q_valid = {q_valid}")

result_valid = jensenshannon(p_valid, q_valid)
print(f"jensenshannon(p_valid, q_valid) = {result_valid}")
print(f"Result is approximately 0: {np.isclose(result_valid, 0.0)}")
```

<details>

<summary>
RuntimeWarning: invalid value encountered in divide, returns NaN
</summary>
```
Testing jensenshannon with all-zero vectors:
p = [0. 0. 0.]
q = [0. 0. 0.]
jensenshannon(p, q) = nan
Result is NaN: True

==================================================
Testing jensenshannon with valid probability vectors:
p_valid = [0.3 0.3 0.4]
q_valid = [0.3 0.3 0.4]
jensenshannon(p_valid, q_valid) = 0.0
Result is approximately 0: True
```
</details>

## Why This Is A Bug

This violates expected behavior in several critical ways:

1. **Violates the fundamental identity property of distance metrics**: For any probability distribution P, the distance JS(P,P) must equal 0. The Jensen-Shannon divergence is defined as a symmetric distance measure, and the identity property is fundamental to its mathematical definition. Returning NaN for all-zero inputs breaks this contract.

2. **Silent failure with unclear error handling**: The function generates RuntimeWarnings about "invalid value encountered in divide" but continues execution and returns NaN. This violates the principle of fail-fast error handling - the function should either handle this edge case gracefully or raise a clear, informative error.

3. **Documentation promises normalization but fails**: The function docstring explicitly states "This routine will normalize `p` and `q` if they don't sum to 1.0", creating an expectation that the function handles non-standard inputs. However, when the sum is 0, the normalization step (lines 1378-1379) performs 0.0/0.0 = NaN, contradicting the documented behavior.

4. **Inconsistent with SciPy's design principles**: Other distance functions in SciPy typically validate inputs and raise ValueError for invalid cases. The silent NaN return is inconsistent with the library's error handling patterns.

## Relevant Context

The bug occurs at lines 1378-1379 in `/home/npc/.local/lib/python3.13/site-packages/scipy/spatial/distance.py`:

```python
p = p / np.sum(p, axis=axis, keepdims=True)  # 0.0/0.0 = NaN when sum is 0
q = q / np.sum(q, axis=axis, keepdims=True)  # 0.0/0.0 = NaN when sum is 0
```

The function's docstring shows an example: `distance.jensenshannon([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])` returning 0.0, confirming the identity property should hold for valid inputs.

Related GitHub issues suggest this is part of a broader pattern:
- Issue #19436: Reports edge cases where jensenshannon returns inf
- Issue #20083: Reports cases where jensenshannon returns NaN due to numerical precision

While all-zero vectors are not valid probability distributions mathematically, users may encounter them in practical scenarios:
- Empty categories in classification tasks
- Uninitialized or placeholder arrays
- Dynamically computed distributions that may be empty
- Testing and validation code

## Proposed Fix

```diff
--- a/scipy/spatial/distance.py
+++ b/scipy/spatial/distance.py
@@ -1375,8 +1375,14 @@ def jensenshannon(p, q, base=None, *, axis=0, keepdims=False):
     """
     p = np.asarray(p)
     q = np.asarray(q)
-    p = p / np.sum(p, axis=axis, keepdims=True)
-    q = q / np.sum(q, axis=axis, keepdims=True)
+    p_sum = np.sum(p, axis=axis, keepdims=True)
+    q_sum = np.sum(q, axis=axis, keepdims=True)
+
+    if np.any(p_sum == 0) or np.any(q_sum == 0):
+        raise ValueError("Input arrays must have at least one non-zero element to compute Jensen-Shannon distance")
+
+    p = p / p_sum
+    q = q / q_sum
     m = (p + q) / 2.0
     left = rel_entr(p, m)
     right = rel_entr(q, m)
```