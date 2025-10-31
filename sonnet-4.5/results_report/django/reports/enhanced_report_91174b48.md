# Bug Report: django.core.serializers.base.ProgressBar Division by Zero

**Target**: `django.core.serializers.base.ProgressBar`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

ProgressBar.update() crashes with ZeroDivisionError when total_count is 0, which occurs when using Django's serialization with progress_output enabled but with the default object_count=0.

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

# Run the test
if __name__ == "__main__":
    test_progressbar_handles_zero_total_count_gracefully()
```

<details>

<summary>
**Failing input**: `count=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/11/hypo.py", line 15, in <module>
    test_progressbar_handles_zero_total_count_gracefully()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/11/hypo.py", line 7, in test_progressbar_handles_zero_total_count_gracefully
    def test_progressbar_handles_zero_total_count_gracefully(count):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/11/hypo.py", line 11, in test_progressbar_handles_zero_total_count_gracefully
    pb.update(count)
    ~~~~~~~~~^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/core/serializers/base.py", line 59, in update
    perc = count * 100 // self.total_count
           ~~~~~~~~~~~~^^~~~~~~~~~~~~~~~~~
ZeroDivisionError: integer division or modulo by zero
Falsifying example: test_progressbar_handles_zero_total_count_gracefully(
    count=1,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from io import StringIO
from django.core.serializers.base import ProgressBar

# Create a ProgressBar with total_count=0
output = StringIO()
pb = ProgressBar(output, total_count=0)

# Attempt to update the progress bar
# This will trigger a ZeroDivisionError on line 59 of base.py
pb.update(1)
```

<details>

<summary>
ZeroDivisionError at line 59 of base.py
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/11/repo.py", line 10, in <module>
    pb.update(1)
    ~~~~~~~~~^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/core/serializers/base.py", line 59, in update
    perc = count * 100 // self.total_count
           ~~~~~~~~~~~~^^~~~~~~~~~~~~~~~~~
ZeroDivisionError: integer division or modulo by zero
```
</details>

## Why This Is A Bug

This violates expected behavior in multiple ways:

1. **Mathematical Error**: Line 59 in ProgressBar.update() performs `perc = count * 100 // self.total_count` without checking if total_count is zero, causing an unhandled ZeroDivisionError.

2. **Default Parameter Trap**: In Serializer.serialize() (line 93), the `object_count` parameter defaults to 0. This default value is then passed directly to ProgressBar's constructor (line 105), creating a progress bar that will crash on any update.

3. **Realistic Usage Scenario**: Users commonly enable progress output without pre-counting objects in their queryset, expecting the serializer to handle this gracefully. The current implementation crashes when the queryset contains any objects.

4. **Poor Error Handling**: The crash provides no helpful context about what went wrong or how to fix it. Users see only "ZeroDivisionError: integer division or modulo by zero" without indication that they need to provide an object_count.

5. **Undocumented Requirement**: Neither the ProgressBar class nor the Serializer.serialize() method documents that object_count must be non-zero when using progress_output. The API accepts the parameter combination but fails at runtime.

## Relevant Context

The bug occurs in Django's serialization framework when using progress bars. The ProgressBar class is an internal utility (no public documentation) used by Django's serializers to display progress during queryset serialization.

The issue manifests when:
- A user calls `serializer.serialize(queryset, progress_output=sys.stderr)`
- They rely on the default `object_count=0` parameter
- The queryset contains at least one object, triggering `progress_bar.update(count)` on line 146 of Serializer.serialize()

This is particularly problematic because:
- Progress bars are often used for large querysets where counting objects beforehand may be expensive
- The default parameter values create a trap for users
- The crash occurs deep in Django's internals, making debugging difficult

Code locations:
- Bug location: `/django/core/serializers/base.py:59` (ProgressBar.update method)
- Default parameter: `/django/core/serializers/base.py:93` (Serializer.serialize method)
- Progress bar instantiation: `/django/core/serializers/base.py:105`
- Update call: `/django/core/serializers/base.py:146`

## Proposed Fix

```diff
--- a/django/core/serializers/base.py
+++ b/django/core/serializers/base.py
@@ -56,6 +56,9 @@ class ProgressBar:
     def update(self, count):
         if not self.output:
             return
+        # Avoid division by zero when total_count is 0
+        if self.total_count == 0:
+            return
         perc = count * 100 // self.total_count
         done = perc * self.progress_width // 100
         if self.prev_done >= done:
```