# Bug Report: ProgressBar Division by Zero

**Target**: `django.core.serializers.base.ProgressBar`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

ProgressBar.update() crashes with ZeroDivisionError when initialized with total_count=0 and update() is called with any positive count value.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.core.serializers.base import ProgressBar
from io import StringIO

@given(st.integers(min_value=1, max_value=100))
def test_progressbar_handles_zero_total_count(count):
    output = StringIO()
    pb = ProgressBar(output, total_count=0)
    pb.update(count)
```

**Failing input**: `count=1` (any positive integer triggers the bug)

## Reproducing the Bug

```python
from django.core.serializers.base import ProgressBar
from io import StringIO

output = StringIO()
pb = ProgressBar(output, total_count=0)
pb.update(1)
```

**Output:**
```
ZeroDivisionError: integer division or modulo by zero
```

## Why This Is A Bug

The ProgressBar class is used during Django serialization to show progress. When serializing an empty queryset with `progress_output` specified and `object_count=0` (the default), the ProgressBar is initialized with `total_count=0`. When `update()` is called during the serialization loop, it performs:

```python
perc = count * 100 // self.total_count  # Line 59 in base.py
```

This causes a ZeroDivisionError when `total_count` is 0. The bug violates the expected behavior that serialization should handle empty querysets gracefully, especially since `object_count` defaults to 0.

## Fix

```diff
--- a/django/core/serializers/base.py
+++ b/django/core/serializers/base.py
@@ -56,6 +56,8 @@ class ProgressBar:
     def update(self, count):
         if not self.output:
             return
+        if self.total_count == 0:
+            return
         perc = count * 100 // self.total_count
         done = perc * self.progress_width // 100
         if self.prev_done >= done:
```