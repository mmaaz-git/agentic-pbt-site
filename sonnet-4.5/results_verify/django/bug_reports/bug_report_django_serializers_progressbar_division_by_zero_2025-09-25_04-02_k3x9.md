# Bug Report: django.core.serializers.base.ProgressBar Division by Zero

**Target**: `django.core.serializers.base.ProgressBar`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

ProgressBar.update() crashes with ZeroDivisionError when total_count is 0, which occurs when serializing empty querysets with progress output enabled.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import io
from django.core.serializers.base import ProgressBar

@given(st.integers(min_value=0, max_value=100))
def test_progressbar_division_by_zero(count):
    output = io.StringIO()
    total_count = 0
    pb = ProgressBar(output, total_count)
    pb.update(count)
```

**Failing input**: `count=0` (or any value when total_count=0)

## Reproducing the Bug

```python
import io
from django.core.serializers.base import ProgressBar

output = io.StringIO()
pb = ProgressBar(output, total_count=0)
pb.update(0)
```

## Why This Is A Bug

The dumpdata management command initializes object_count to 0 (line 231 in dumpdata.py) and can remain 0 when serializing an empty database or when all models are filtered out. When progress_output is enabled and object_count is 0, the ProgressBar crashes on the first update() call due to division by zero at line 59 in base.py:

```python
perc = count * 100 // self.total_count
```

This makes the dumpdata command unusable for empty databases when outputting to a file with verbose mode.

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