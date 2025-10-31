# Bug Report: scipy.stats.quantile Unnecessarily Rejects Integer p Values

**Target**: `scipy.stats.quantile`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`scipy.stats.quantile` raises a `ValueError` when passed integer values for the probability parameter `p`, even though integers 0 and 1 are mathematically valid probability values and commonly used to get minimum and maximum values.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
from scipy import stats

@given(
    data=st.lists(
        st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
        min_size=5,
        max_size=100
    )
)
def test_quantile_zero_is_min(data):
    x = np.array(data)
    q0 = stats.quantile(x, 0)
    min_x = np.min(x)
    assert np.allclose(q0, min_x), f"quantile(x, 0) should be min(x)"

if __name__ == "__main__":
    test_quantile_zero_is_min()
```

<details>

<summary>
**Failing input**: `data=[0.0, 0.0, 0.0, 0.0, 0.0]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/16/hypo.py", line 19, in <module>
    test_quantile_zero_is_min()
    ~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/16/hypo.py", line 6, in test_quantile_zero_is_min
    data=st.lists(
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/16/hypo.py", line 14, in test_quantile_zero_is_min
    q0 = stats.quantile(x, 0)
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/stats/_quantile.py", line 269, in quantile
    temp = _quantile_iv(x, p, method, axis, nan_policy, keepdims)
  File "/home/npc/.local/lib/python3.13/site-packages/scipy/stats/_quantile.py", line 16, in _quantile_iv
    raise ValueError("`p` must have real floating dtype.")
ValueError: `p` must have real floating dtype.
Falsifying example: test_quantile_zero_is_min(
    data=[0.0, 0.0, 0.0, 0.0, 0.0],
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
from scipy import stats

# Create test data
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

print("Testing scipy.stats.quantile with integer p values:")
print("=" * 60)
print("Test data x =", x)
print()

# Test 1: Integer 0 (should fail)
print("Test 1: stats.quantile(x, 0) with integer 0")
print("-" * 40)
try:
    result = stats.quantile(x, 0)
    print(f"Success: {result}")
except ValueError as e:
    print(f"ValueError: {e}")

print()

# Test 2: Float 0.0 (should work)
print("Test 2: stats.quantile(x, 0.0) with float 0.0")
print("-" * 40)
try:
    result = stats.quantile(x, 0.0)
    print(f"Success: {result}")
except ValueError as e:
    print(f"ValueError: {e}")

print()

# Test 3: Integer 1 (should fail)
print("Test 3: stats.quantile(x, 1) with integer 1")
print("-" * 40)
try:
    result = stats.quantile(x, 1)
    print(f"Success: {result}")
except ValueError as e:
    print(f"ValueError: {e}")

print()

# Test 4: Float 1.0 (should work)
print("Test 4: stats.quantile(x, 1.0) with float 1.0")
print("-" * 40)
try:
    result = stats.quantile(x, 1.0)
    print(f"Success: {result}")
except ValueError as e:
    print(f"ValueError: {e}")

print()

# Compare with NumPy's behavior
print("Comparison with NumPy functions:")
print("=" * 60)

# Test NumPy quantile
print("np.quantile(x, 0) =", np.quantile(x, 0))
print("np.quantile(x, 1) =", np.quantile(x, 1))
print("np.percentile(x, 0) =", np.percentile(x, 0))
print("np.percentile(x, 100) =", np.percentile(x, 100))

print()
print("Conclusion: NumPy accepts integer p values without issue,")
print("while scipy.stats.quantile unnecessarily rejects them.")
```

<details>

<summary>
ValueError when using integer p values
</summary>
```
Testing scipy.stats.quantile with integer p values:
============================================================
Test data x = [1. 2. 3. 4. 5.]

Test 1: stats.quantile(x, 0) with integer 0
----------------------------------------
ValueError: `p` must have real floating dtype.

Test 2: stats.quantile(x, 0.0) with float 0.0
----------------------------------------
Success: 1.0

Test 3: stats.quantile(x, 1) with integer 1
----------------------------------------
ValueError: `p` must have real floating dtype.

Test 4: stats.quantile(x, 1.0) with float 1.0
----------------------------------------
Success: 5.0

Comparison with NumPy functions:
============================================================
np.quantile(x, 0) = 1.0
np.quantile(x, 1) = 5.0
np.percentile(x, 0) = 1.0
np.percentile(x, 100) = 5.0

Conclusion: NumPy accepts integer p values without issue,
while scipy.stats.quantile unnecessarily rejects them.
```
</details>

## Why This Is A Bug

This violates expected behavior for several critical reasons:

1. **Mathematical Validity**: The values 0 and 1 are valid probabilities for quantile computation. There is no mathematical distinction between the integer 0 and the float 0.0 - they both represent the same probability value. The quantile function is mathematically defined for any p where 0 ≤ p ≤ 1, regardless of the numeric type used to represent p.

2. **API Inconsistency with NumPy**: NumPy's `quantile` and `percentile` functions accept integer p values without issue. Since SciPy and NumPy are part of the same scientific computing ecosystem, users expect consistent behavior. This inconsistency creates friction when switching between the two libraries.

3. **Violation of Python Conventions**: In Python, numeric types are generally interchangeable when the values are mathematically equivalent. Most NumPy/SciPy functions automatically coerce compatible numeric types. Rejecting integers violates this principle of least surprise.

4. **Common Use Case Impact**: Using `quantile(x, 0)` to get the minimum and `quantile(x, 1)` to get the maximum are extremely common operations. Integer literals 0 and 1 are the most natural way to express these boundary values in Python code.

5. **Documentation Ambiguity**: The documentation states that p should be "array_like of float" with "Values must be between 0 and 1 (inclusive)" but doesn't explicitly state that the dtype must be floating-point. The emphasis is on the value range, not the type. The error message "`p` must have real floating dtype`" reveals an implementation detail that shouldn't be exposed to users.

6. **Unnecessary Restriction**: The code already contains the machinery to handle this conversion. Line 18 of `_quantile_iv` calls `xp_promote(x, p, force_floating=True, xp=xp)` which would automatically convert integer p to float if the initial type check were removed or relaxed.

## Relevant Context

The bug occurs in `/scipy/stats/_quantile.py` at line 15-16 in the `_quantile_iv` function:

```python
if not xp.isdtype(xp.asarray(p).dtype, 'real floating'):
    raise ValueError("`p` must have real floating dtype.")
```

This check explicitly rejects integral types even though:
- Line 12 accepts both 'integral' and 'real floating' for x
- Line 18 immediately calls `xp_promote` with `force_floating=True` which would handle the conversion
- The check appears inconsistent: it accepts integral types for x but not for p

Documentation link: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.quantile.html
Source code: https://github.com/scipy/scipy/blob/main/scipy/stats/_quantile.py

The issue affects all methods ('linear', 'inverted_cdf', 'harrell-davis', etc.) since the type check occurs before method-specific code.

## Proposed Fix

The fix is straightforward - modify the type check in `_quantile_iv` to accept both integral and real floating dtypes, similar to how x is handled:

```diff
--- a/scipy/stats/_quantile.py
+++ b/scipy/stats/_quantile.py
@@ -12,8 +12,8 @@ def _quantile_iv(x, p, method, axis, nan_policy, keepdims):
     if not xp.isdtype(xp.asarray(x).dtype, ('integral', 'real floating')):
         raise ValueError("`x` must have real dtype.")

-    if not xp.isdtype(xp.asarray(p).dtype, 'real floating'):
-        raise ValueError("`p` must have real floating dtype.")
+    if not xp.isdtype(xp.asarray(p).dtype, ('integral', 'real floating')):
+        raise ValueError("`p` must have real dtype.")

     x, p = xp_promote(x, p, force_floating=True, xp=xp)
     dtype = x.dtype
```

This minimal change leverages the existing `xp_promote` call on line 18 which already handles the conversion to float with `force_floating=True`. The fix is non-breaking since it only accepts more inputs, not fewer.