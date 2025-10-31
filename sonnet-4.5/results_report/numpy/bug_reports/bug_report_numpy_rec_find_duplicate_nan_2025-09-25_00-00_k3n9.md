# Bug Report: numpy.rec.find_duplicate NaN Detection Failure

**Target**: `numpy.rec.find_duplicate`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.rec.find_duplicate` fails to detect duplicate NaN values in a list, returning an empty result even when multiple NaN values are present.

## Property-Based Test

```python
import numpy.rec
import math
from hypothesis import given, strategies as st


@given(st.lists(st.floats(allow_nan=True, allow_infinity=False), min_size=0, max_size=20))
def test_find_duplicate_nan_behavior(lst):
    result = numpy.rec.find_duplicate(lst)
    nan_count = sum(1 for x in lst if isinstance(x, float) and math.isnan(x))

    if nan_count > 1:
        nan_in_result = any(isinstance(x, float) and math.isnan(x) for x in result)
        assert nan_in_result, f"Multiple NaN values in input but none in result"
```

**Failing input**: `[nan, nan, 1.0, 1.0]`

## Reproducing the Bug

```python
import numpy.rec

lst = [float('nan'), float('nan'), 1.0, 1.0]
result = numpy.rec.find_duplicate(lst)

print(f"Input: {lst}")
print(f"Result: {result}")

assert len(result) == 2
```

Output:
```
Input: [nan, nan, 1.0, 1.0]
Result: [1.0]
AssertionError
```

## Why This Is A Bug

The function is documented to "Find duplication in a list, return a list of duplicated elements". When a list contains multiple NaN values (e.g., two or more), these are duplicated elements that should be reported. However, the current implementation uses Python's `Counter`, which treats each NaN as a distinct value because `nan != nan` in Python. This causes the function to count each NaN separately (each with count=1) and fail to identify them as duplicates.

This violates user expectations for a duplicate-finding function, especially in scientific computing contexts where NaN values are common and users would reasonably expect duplicate NaN values to be detected.

## Fix

```diff
--- a/numpy/_core/records.py
+++ b/numpy/_core/records.py
@@ -1,8 +1,17 @@
 @set_module('numpy.rec')
 def find_duplicate(list):
     """Find duplication in a list, return a list of duplicated elements"""
-    return [
-        item
-        for item, counts in Counter(list).items()
-        if counts > 1
-    ]
+    import math
+
+    nan_count = sum(1 for x in list if isinstance(x, float) and math.isnan(x))
+
+    counter_result = [
+        item
+        for item, counts in Counter(list).items()
+        if counts > 1 and not (isinstance(item, float) and math.isnan(item))
+    ]
+
+    if nan_count > 1:
+        counter_result.append(float('nan'))
+
+    return counter_result
```