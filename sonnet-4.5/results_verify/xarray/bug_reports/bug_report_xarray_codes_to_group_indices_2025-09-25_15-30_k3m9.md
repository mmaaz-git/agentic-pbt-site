# Bug Report: xarray.core.groupby._codes_to_group_indices IndexError on Invalid Code Values

**Target**: `xarray.core.groupby._codes_to_group_indices`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `_codes_to_group_indices` function crashes with an `IndexError` when code values are >= N (the number of groups), instead of either validating inputs or providing a descriptive error message.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, settings, strategies as st
from xarray.core.groupby import _codes_to_group_indices

@given(
    codes=st.lists(st.integers(min_value=-5, max_value=10), min_size=0, max_size=100),
    N=st.integers(min_value=1, max_value=20)
)
@settings(max_examples=500)
def test_codes_to_group_indices_all_indices_present(codes, N):
    """All non-negative indices should appear exactly once in result."""
    codes_arr = np.array(codes, dtype=np.int64)
    result = _codes_to_group_indices(codes_arr, N)

    all_indices = []
    for group in result:
        all_indices.extend(group)

    expected_indices = [i for i, c in enumerate(codes) if c >= 0]

    assert sorted(all_indices) == sorted(expected_indices)
```

**Failing input**: `codes=[1], N=1` (and many others)

## Reproducing the Bug

```python
import numpy as np
from xarray.core.groupby import _codes_to_group_indices

codes = np.array([1], dtype=np.int64)
result = _codes_to_group_indices(codes, N=1)
```

**Output:**
```
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "xarray/core/groupby.py", line 92, in _codes_to_group_indices
    groups[g].append(n)
    ^^^^^^^^^
IndexError: tuple index out of range
```

**Additional examples:**
```python
codes = np.array([5, 2, 8], dtype=np.int64)
result = _codes_to_group_indices(codes, N=3)

codes = np.array([0, 1, 2, 3], dtype=np.int64)
result = _codes_to_group_indices(codes, N=2)
```

Both raise `IndexError: tuple index out of range`

## Why This Is A Bug

1. **Precondition Violation**: The function has an implicit precondition that all codes must be in the range `[-1, N)` (where -1 indicates "no group"), but this precondition is not documented or validated.

2. **Crash on Invalid Input**: When this precondition is violated, the function crashes with a cryptic `IndexError` instead of providing a meaningful error message.

3. **Inconsistent Validation**: The function validates that codes are 1-dimensional (`assert codes.ndim == 1`) and handles negative codes gracefully (`if g >= 0`), suggesting that input validation is expected. However, it fails to validate the upper bound.

4. **API Fragility**: Internal functions that don't validate their inputs are fragile and can lead to hard-to-debug issues when the calling code has bugs. A descriptive error message would help developers identify the issue quickly.

## Impact Assessment

**Severity: Medium** because:
- This is an internal function (prefixed with `_`), so direct user impact is limited
- However, bugs in calling code can cause cryptic crashes
- The fix is simple and improves code robustness

## Fix

Add validation to check that all codes are within the valid range:

```diff
diff --git a/xarray/core/groupby.py b/xarray/core/groupby.py
index 1234567..abcdefg 100644
--- a/xarray/core/groupby.py
+++ b/xarray/core/groupby.py
@@ -85,10 +85,16 @@ def _codes_to_group_indices(codes: np.ndarray, N: int) -> GroupIndices:

 def _codes_to_group_indices(codes: np.ndarray, N: int) -> GroupIndices:
     """Converts integer codes for groups to group indices."""
     assert codes.ndim == 1
     groups: GroupIndices = tuple([] for _ in range(N))
     for n, g in enumerate(codes):
         if g >= 0:
+            if g >= N:
+                raise ValueError(
+                    f"Code value {g} at index {n} is out of range. "
+                    f"Expected codes in range [-1, {N}), got {g}."
+                )
             groups[g].append(n)
     return groups
```

Alternative approach (fail fast):
```diff
@@ -85,10 +85,15 @@ def _codes_to_group_indices(codes: np.ndarray, N: int) -> GroupIndices:

 def _codes_to_group_indices(codes: np.ndarray, N: int) -> GroupIndices:
     """Converts integer codes for groups to group indices."""
     assert codes.ndim == 1
+    max_code = codes.max() if len(codes) > 0 else -1
+    if max_code >= N:
+        raise ValueError(
+            f"Maximum code value {max_code} exceeds N={N}. "
+            f"All codes must be in range [-1, {N})."
+        )
     groups: GroupIndices = tuple([] for _ in range(N))
     for n, g in enumerate(codes):
         if g >= 0:
             groups[g].append(n)
     return groups
```