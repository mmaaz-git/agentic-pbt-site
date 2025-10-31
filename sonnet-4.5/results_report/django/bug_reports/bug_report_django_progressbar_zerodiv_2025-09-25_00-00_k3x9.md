# Bug Report: django.core.serializers.base.ProgressBar ZeroDivisionError

**Target**: `django.core.serializers.base.ProgressBar.update`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

ProgressBar.update() crashes with ZeroDivisionError when total_count is 0, which occurs when running `dumpdata` on an empty database with progress output enabled.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from django.core.serializers.base import ProgressBar
from io import StringIO

@given(st.integers(min_value=0, max_value=1000), st.integers(min_value=0, max_value=1000))
def test_progress_bar_no_crash(total_count, count):
    assume(count <= total_count)
    output = StringIO()
    pb = ProgressBar(output, total_count)
    pb.update(count)
```

**Failing input**: `total_count=0, count=0`

## Reproducing the Bug

```python
from io import StringIO
from django.core.serializers.base import ProgressBar

output = StringIO()
pb = ProgressBar(output, total_count=0)
pb.update(0)
```

## Why This Is A Bug

The bug occurs in real usage when running `django-admin dumpdata` on an empty database with progress output enabled (e.g., dumping to a file with verbosity > 0 on a TTY). The dumpdata command calculates `object_count = sum(get_objects(count_only=True))` which can be 0, then passes it to `serializers.serialize(..., object_count=object_count)`, which creates `ProgressBar(progress_output, 0)`. When `update(0)` is called, line 59 in base.py crashes: `perc = count * 100 // self.total_count` → ZeroDivisionError.

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