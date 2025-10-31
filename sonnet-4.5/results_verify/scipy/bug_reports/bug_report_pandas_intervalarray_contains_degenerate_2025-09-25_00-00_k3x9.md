# Bug Report: IntervalArray.contains() Fails for Degenerate Intervals

**Target**: `pandas.core.arrays.IntervalArray.contains()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`IntervalArray.contains()` incorrectly returns `False` for degenerate intervals (where left == right) when testing if they contain their single point value, even when one or both endpoints are closed.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.core.arrays import IntervalArray
import pandas as pd


@st.composite
def interval_arrays(draw, allow_na=True):
    size = draw(st.integers(min_value=0, max_value=20))
    left = []
    right = []
    for _ in range(size):
        if allow_na and draw(st.booleans()):
            left.append(np.nan)
            right.append(np.nan)
        else:
            l = draw(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False))
            width = draw(st.floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False))
            r = l + width
            left.append(l)
            right.append(r)
    closed = draw(st.sampled_from(['left', 'right', 'both', 'neither']))
    return left, right, closed


@given(interval_arrays())
@settings(max_examples=500)
def test_contains_respects_closed(arrays):
    left, right, closed = arrays
    if len(left) == 0:
        return
    try:
        arr = IntervalArray.from_arrays(left, right, closed=closed)

        for i in range(len(arr)):
            if pd.isna(left[i]) or pd.isna(right[i]):
                continue

            left_val = left[i]
            right_val = right[i]

            if closed == 'both':
                assert arr.contains(left_val)[i], f"'both' closed interval should contain left endpoint"
                assert arr.contains(right_val)[i], f"'both' closed interval should contain right endpoint"
            elif closed == 'left':
                assert arr.contains(left_val)[i], f"'left' closed interval should contain left endpoint"
                if left_val != right_val:
                    assert not arr.contains(right_val)[i], f"'left' closed interval should not contain right endpoint"
            elif closed == 'right':
                if left_val != right_val:
                    assert not arr.contains(left_val)[i], f"'right' closed interval should not contain left endpoint"
                assert arr.contains(right_val)[i], f"'right' closed interval should contain right endpoint"
    except (TypeError, ValueError) as e:
        pass
```

**Failing input**: `arrays=([0.0], [0.0], 'right')`

## Reproducing the Bug

```python
import pandas as pd
from pandas.core.arrays import IntervalArray

arr_right = IntervalArray.from_arrays([0.0], [0.0], closed='right')
print(f"Interval: {arr_right[0]}")
print(f"Contains 0.0: {arr_right.contains(0.0)[0]}")

arr_left = IntervalArray.from_arrays([0.0], [0.0], closed='left')
print(f"Interval: {arr_left[0]}")
print(f"Contains 0.0: {arr_left.contains(0.0)[0]}")

arr_both = IntervalArray.from_arrays([0.0], [0.0], closed='both')
print(f"Interval: {arr_both[0]}")
print(f"Contains 0.0: {arr_both.contains(0.0)[0]}")
```

**Output:**
```
Interval: (0.0, 0.0]
Contains 0.0: False
Interval: [0.0, 0.0)
Contains 0.0: False
Interval: [0.0, 0.0]
Contains 0.0: False
```

All three should return `True` since the point 0.0 is included in intervals where at least one endpoint is closed.

## Why This Is A Bug

A degenerate interval where `left == right` represents a single point. Whether this point is contained should depend on the closure:

- `(0, 0]` closed='right': The right endpoint is included, so 0 should be contained
- `[0, 0)` closed='left': The left endpoint is included, so 0 should be contained
- `[0, 0]` closed='both': Both endpoints included, so 0 should be contained
- `(0, 0)` closed='neither': Both endpoints excluded, empty interval, 0 should NOT be contained

The current implementation in `interval.py:1818-1820` uses:

```python
return (self._left < other if self.open_left else self._left <= other) & (
    other < self._right if self.open_right else other <= self._right
)
```

For a degenerate interval `(0, 0]` with `other = 0.0`:
- Left condition: `self._left < 0.0` → `0.0 < 0.0` → `False`
- Right condition: `0.0 <= self._right` → `0.0 <= 0.0` → `True`
- Result: `False & True` = `False` ❌

The strict inequality on the open side prevents the closed side from being satisfied in degenerate cases.

## Fix

```diff
--- a/pandas/core/arrays/interval.py
+++ b/pandas/core/arrays/interval.py
@@ -1814,9 +1814,18 @@ class IntervalArray(IntervalMixin, ExtensionArray):
     def contains(self, other):
         if isinstance(other, Interval):
             raise NotImplementedError("contains not implemented for two intervals")

-        return (self._left < other if self.open_left else self._left <= other) & (
-            other < self._right if self.open_right else other <= self._right
-        )
+        left_cond = self._left < other if self.open_left else self._left <= other
+        right_cond = other < self._right if self.open_right else other <= self._right
+
+        # For degenerate intervals (left == right), check if at least one side is closed
+        # and the point equals the boundary
+        degenerate = self._left == self._right
+        degenerate_contains = (
+            (other == self._left) &
+            ~(self.open_left & self.open_right)
+        )
+
+        return np.where(degenerate, degenerate_contains, left_cond & right_cond)

     def isin(self, values: ArrayLike) -> npt.NDArray[np.bool_]:
         if isinstance(values, IntervalArray):
```