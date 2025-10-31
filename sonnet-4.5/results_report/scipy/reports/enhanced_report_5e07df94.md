# Bug Report: django.core.serializers ProgressBar Division by Zero

**Target**: `django.core.serializers.base.ProgressBar`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The ProgressBar class in Django's serializer module crashes with a ZeroDivisionError when initialized with total_count=0 and update() is called, due to unchecked division on line 59.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'test_settings')

from hypothesis import given, strategies as st
from io import StringIO
from django.core.serializers.base import ProgressBar

@given(st.integers(min_value=0, max_value=0))
def test_progressbar_zero_total_count(total_count):
    output = StringIO()
    progress = ProgressBar(output, total_count)
    progress.update(0)

# Run the test
test_progressbar_zero_total_count()
```

<details>

<summary>
**Failing input**: `total_count=0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/46/hypo.py", line 17, in <module>
    test_progressbar_zero_total_count()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/46/hypo.py", line 11, in test_progressbar_zero_total_count
    def test_progressbar_zero_total_count(total_count):
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/46/hypo.py", line 14, in test_progressbar_zero_total_count
    progress.update(0)
    ~~~~~~~~~~~~~~~^^^
  File "/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/core/serializers/base.py", line 59, in update
    perc = count * 100 // self.total_count
           ~~~~~~~~~~~~^^~~~~~~~~~~~~~~~~~
ZeroDivisionError: integer division or modulo by zero
Falsifying example: test_progressbar_zero_total_count(
    total_count=0,
)
```
</details>

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

<details>

<summary>
ZeroDivisionError when calling update() on ProgressBar with total_count=0
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/46/repo.py", line 11, in <module>
    progress.update(0)
    ~~~~~~~~~~~~~~~^^^
  File "/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/core/serializers/base.py", line 59, in update
    perc = count * 100 // self.total_count
           ~~~~~~~~~~~~^^~~~~~~~~~~~~~~~~~
ZeroDivisionError: integer division or modulo by zero
```
</details>

## Why This Is A Bug

This is a legitimate programming error that violates basic defensive programming principles. The ProgressBar.update() method performs division by self.total_count on line 59 without checking if it's zero, causing an unhandled ZeroDivisionError. While this bug doesn't manifest in normal Django serialization operations (due to how the serializer loop is structured with `enumerate(queryset, start=1)` on line 109), the ProgressBar class itself should be robust enough to handle edge cases gracefully.

Progress bars are commonly expected to handle empty collections without crashing. Even though ProgressBar is an internal class not intended for public use, properly written code should anticipate and handle division by zero cases. The fact that the current Django serialization code path doesn't trigger this bug is fortunate but doesn't excuse the underlying error in the ProgressBar implementation.

## Relevant Context

The bug is located in `/django/core/serializers/base.py` at line 59 of the ProgressBar class. The class is used internally by Django's serialization framework to display progress when serializing querysets with the `progress_output` parameter enabled.

In the normal Django serialization flow (lines 105-146 of the Serializer.serialize method):
- Line 105: A ProgressBar is created with `object_count` as the total_count
- Line 109: The loop iterates with `enumerate(queryset, start=1)`
- Line 146: `progress_bar.update(count)` is called inside the loop
- With an empty queryset (object_count=0), the loop body never executes, so update() is never called

However, the ProgressBar class itself lacks proper input validation, making it vulnerable to crashes when used directly or if Django's serialization logic changes in the future.

Django source code reference: https://github.com/django/django/blob/main/django/core/serializers/base.py

## Proposed Fix

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