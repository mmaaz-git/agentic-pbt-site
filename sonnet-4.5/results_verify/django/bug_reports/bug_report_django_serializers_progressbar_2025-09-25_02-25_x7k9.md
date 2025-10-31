# Bug Report: django.core.serializers.base.ProgressBar Division by Zero

**Target**: `django.core.serializers.base.ProgressBar`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `ProgressBar.update()` method crashes with `ZeroDivisionError` when `total_count` is initialized to 0, which is the default value for the `object_count` parameter in `Serializer.serialize()`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from io import StringIO
from django.core.serializers.base import ProgressBar


@given(count=st.integers(min_value=0, max_value=100))
def test_progressbar_zero_total_count(count):
    output = StringIO()
    progress_bar = ProgressBar(output, total_count=0)
    progress_bar.update(count)
```

**Failing input**: `count=0` (or any non-negative integer)

## Reproducing the Bug

```python
from io import StringIO
from django.core.serializers.base import ProgressBar

output = StringIO()
progress_bar = ProgressBar(output, total_count=0)
progress_bar.update(1)
```

Output:
```
ZeroDivisionError: integer division or modulo by zero
```

## Why This Is A Bug

The `ProgressBar` class is initialized with `total_count` from the `object_count` parameter in `Serializer.serialize()`, which defaults to 0 (base.py:93). When a progress bar is created with `total_count=0` and then `update()` is called with any count, it performs `count * 100 // self.total_count` (base.py:59), resulting in a division by zero error.

This can happen when:
1. Users don't specify `object_count` when calling `serialize()`
2. Users explicitly pass `object_count=0`
3. The actual count of objects to serialize is unknown

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

Alternatively, check at initialization:

```diff
--- a/django/core/serializers/base.py
+++ b/django/core/serializers/base.py
@@ -50,6 +50,7 @@ class ProgressBar:

     def __init__(self, output, total_count):
         self.output = output
+        self.total_count = max(1, total_count)  # Avoid division by zero
-        self.total_count = total_count
         self.prev_done = 0
```