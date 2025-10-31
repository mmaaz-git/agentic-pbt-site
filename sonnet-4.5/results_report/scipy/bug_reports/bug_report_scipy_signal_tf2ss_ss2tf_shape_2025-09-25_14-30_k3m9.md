# Bug Report: scipy.signal tf2ss/ss2tf Shape Inconsistency

**Target**: `scipy.signal.tf2ss` and `scipy.signal.ss2tf`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The functions `tf2ss` and `ss2tf` have incompatible input/output shapes, making round-trip conversions between transfer function and state-space representations impossible without manual shape manipulation. `tf2ss` accepts 1-D arrays for numerator coefficients, but `ss2tf` returns a 2-D array, preventing direct round-trip conversion.

## Property-Based Test

```python
import numpy as np
import scipy.signal
from hypothesis import given, strategies as st, settings

@settings(max_examples=200)
@given(
    num=st.lists(st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False), min_size=1, max_size=5),
    den=st.lists(st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False), min_size=1, max_size=5)
)
def test_tf2ss_ss2tf_shape_consistency(num, den):
    from hypothesis import assume
    assume(abs(den[0]) > 0.1)
    assume(len(den) >= len(num))

    num_arr = np.array(num)
    den_arr = np.array(den)

    A, B, C, D = scipy.signal.tf2ss(num_arr, den_arr)
    num_recovered, den_recovered = scipy.signal.ss2tf(A, B, C, D)

    assert num_arr.ndim == 1, f"Input num should be 1D, got {num_arr.ndim}D"
    assert den_arr.ndim == 1, f"Input den should be 1D, got {den_arr.ndim}D"

    assert num_recovered.ndim == num_arr.ndim, \
        f"Shape inconsistency: input num has {num_arr.ndim} dimensions, but recovered has {num_recovered.ndim} dimensions"
```

**Failing input**: `num=[0.0], den=[1.0]` (or any other valid transfer function coefficients)

## Reproducing the Bug

```python
import numpy as np
import scipy.signal

num = np.array([1.0, 2.0])
den = np.array([1.0, 3.0, 4.0])

A, B, C, D = scipy.signal.tf2ss(num, den)
num_recovered, den_recovered = scipy.signal.ss2tf(A, B, C, D)

print(f"Input num shape: {num.shape}")
print(f"Output num shape: {num_recovered.shape}")

try:
    A2, B2, C2, D2 = scipy.signal.tf2ss(num_recovered, den_recovered)
except Exception as e:
    print(f"Error on round-trip: {e}")
```

## Why This Is A Bug

This violates the expected contract for conversion functions. When two functions are designed to convert between representations (like `tf2ss` and `ss2tf`), users reasonably expect that:

1. Converting from A to B and back to A should work without manual intervention
2. The output shape should be compatible with the input shape for round-trip operations

The current behavior breaks this expectation:
- `tf2ss` accepts `num` as a 1-D array: `[1.0, 2.0]`
- `ss2tf` returns `num` as a 2-D array: `[[0.0, 1.0, 2.0]]`
- Passing the 2-D output back to `tf2ss` would require manual `squeeze()` operation

According to the documentation, `ss2tf` returns a 2-D array "with one row for each of the system's outputs." However, for single-output systems (the common case), this creates unnecessary friction and breaks the round-trip conversion pattern.

## Fix

For single-output systems (q=1), `ss2tf` should return a 1-D array for `num` to maintain compatibility with `tf2ss`. This can be done by automatically squeezing the result when there is only one output:

```diff
--- a/scipy/signal/_lti_conversion.py
+++ b/scipy/signal/_lti_conversion.py
@@ -somewhere in ss2tf function
     # Current behavior: always returns 2-D num
     num = ...  # 2-D array with shape (q, n)

+    # For single-output systems, squeeze to 1-D for compatibility with tf2ss
+    if num.shape[0] == 1:
+        num = num.squeeze(axis=0)
+
     return num, den
```

Alternatively, the documentation could be updated to explicitly warn users that round-trip conversion requires manual shape manipulation, though this is a worse user experience.