# Bug Report: django.core.serializers ProgressBar Division by Zero

**Target**: `django.core.serializers.base.ProgressBar`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

ProgressBar crashes with ZeroDivisionError when total_count is 0 and update() is called, which happens when serializing empty querysets with progress_output enabled.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from io import StringIO
from django.core.serializers.base import ProgressBar

@given(st.integers(min_value=0, max_value=0))
def test_progressbar_zero_total_count(total_count):
    output = StringIO()
    progress = ProgressBar(output, total_count)
    progress.update(0)
```

**Failing input**: `total_count=0`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'test_settings')

from io import StringIO
from django.core.serializers.base import ProgressBar

output = StringIO()
progress = ProgressBar(output, total_count=0)
progress.update(0)
```

Output:
```
ZeroDivisionError: integer division or modulo by zero
  File "django/core/serializers/base.py", line 59, in update
    perc = count * 100 // self.total_count
```

## Why This Is A Bug

When serializing an empty queryset with `progress_output` enabled, the ProgressBar is created with `object_count=0` (line 105 in base.py). The first call to `progress_bar.update(count)` (line 146) triggers a division by zero on line 59.

This violates reasonable expectations: progress bars should handle empty collections gracefully, not crash.

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