# Bug Report: django.core.serializers.base.ProgressBar Division by Zero

**Target**: `django.core.serializers.base.ProgressBar`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

ProgressBar.update() crashes with ZeroDivisionError when total_count is 0, which can occur when serializing with progress_output but without specifying object_count.

## Property-Based Test

```python
from io import StringIO
from django.core.serializers.base import ProgressBar
from hypothesis import given, strategies as st


@given(st.integers(min_value=1, max_value=1000))
def test_progressbar_handles_zero_total_count_gracefully(count):
    output = StringIO()
    pb = ProgressBar(output, total_count=0)

    pb.update(count)
```

**Failing input**: `count=1`

## Reproducing the Bug

```python
from io import StringIO
from django.core.serializers.base import ProgressBar

output = StringIO()
pb = ProgressBar(output, total_count=0)
pb.update(1)
```

Output:
```
ZeroDivisionError: integer division or modulo by zero
```

## Why This Is A Bug

1. In `Serializer.serialize()`, the `object_count` parameter defaults to 0
2. ProgressBar is created with `progress_bar = self.progress_class(progress_output, object_count)`
3. If a user enables progress output but doesn't specify object_count, ProgressBar receives total_count=0
4. When the queryset has objects, update() is called which executes `perc = count * 100 // self.total_count`
5. This triggers ZeroDivisionError when total_count=0

This is a realistic scenario where users enable progress bars without explicitly counting objects first.

## Fix

```diff
--- a/django/core/serializers/base.py
+++ b/django/core/serializers/base.py
@@ -44,6 +44,8 @@ class ProgressBar:

     def update(self, count):
         if not self.output:
             return
+        if self.total_count == 0:
+            return
         perc = count * 100 // self.total_count
         done = perc * self.progress_width // 100
         if self.prev_done >= done:
```

Alternatively, clamp the done value to prevent memory issues when count > total_count:

```diff
--- a/django/core/serializers/base.py
+++ b/django/core/serializers/base.py
@@ -44,9 +44,12 @@ class ProgressBar:

     def update(self, count):
         if not self.output:
             return
+        if self.total_count == 0:
+            return
         perc = count * 100 // self.total_count
         done = perc * self.progress_width // 100
+        done = min(done, self.progress_width)
         if self.prev_done >= done:
             return
         self.prev_done = done
```