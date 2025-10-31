# Bug Report: django.core.serializers.base.ProgressBar Division by Zero

**Target**: `django.core.serializers.base.ProgressBar`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

ProgressBar.update() crashes with ZeroDivisionError when total_count is 0, which occurs when Django's dumpdata command attempts to serialize an empty database with progress output enabled.

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

# Run the test
test_progressbar_division_by_zero()
```

<details>

<summary>
**Failing input**: `count=0` (or any value when total_count=0)
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 13, in <module>
    test_progressbar_division_by_zero()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 6, in test_progressbar_division_by_zero
    def test_progressbar_division_by_zero(count):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 10, in test_progressbar_division_by_zero
    pb.update(count)
    ~~~~~~~~~^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/core/serializers/base.py", line 59, in update
    perc = count * 100 // self.total_count
           ~~~~~~~~~~~~^^~~~~~~~~~~~~~~~~~
ZeroDivisionError: integer division or modulo by zero
Falsifying example: test_progressbar_division_by_zero(
    count=0,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import io
from django.core.serializers.base import ProgressBar

# Create a ProgressBar with total_count=0 (simulating an empty database)
output = io.StringIO()
pb = ProgressBar(output, total_count=0)

# Attempt to update the progress bar
# This will cause a ZeroDivisionError at line 59 of base.py
pb.update(0)
```

<details>

<summary>
ZeroDivisionError: integer division or modulo by zero
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/51/repo.py", line 10, in <module>
    pb.update(0)
    ~~~~~~~~~^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/core/serializers/base.py", line 59, in update
    perc = count * 100 // self.total_count
           ~~~~~~~~~~~~^^~~~~~~~~~~~~~~~~~
ZeroDivisionError: integer division or modulo by zero
```
</details>

## Why This Is A Bug

This bug violates the expected behavior of Django's serialization framework in several ways:

1. **Mathematical Error**: The code performs division by zero at line 59 of `/django/core/serializers/base.py` without checking if `total_count` is 0:
   ```python
   perc = count * 100 // self.total_count  # Crashes when self.total_count is 0
   ```

2. **Real-World Impact**: The dumpdata management command initializes `object_count` to 0 (line 235 in `/django/core/management/commands/dumpdata.py`) when serializing an empty database. This value is then passed to ProgressBar's constructor (line 277), causing the crash on the first update() call.

3. **Valid Use Case**: Empty databases are a legitimate scenario that occurs frequently:
   - During initial development when no data has been added yet
   - In test environments with clean databases
   - After running database migrations that create new tables
   - When all models are filtered out by the dumpdata command's filters

4. **No Documentation of Limitation**: The ProgressBar class has no documentation (no class docstring or method docstrings) that indicates `total_count=0` is invalid or unsupported. A reasonable user would expect a progress bar to handle "no items to process" gracefully.

5. **Command Failure**: This bug makes the dumpdata command completely unusable for empty databases when using verbose mode with output to a file, breaking a core Django management command.

## Relevant Context

The bug manifests specifically when using Django's dumpdata management command with these conditions:
- Database has no objects to serialize (or all are filtered out)
- Output is directed to a file (not stdout)
- Verbose mode is enabled (verbosity > 0)
- Terminal is interactive (isatty() returns True)

Code flow:
1. `dumpdata.py:235` - Calculates `object_count` by summing counts, resulting in 0 for empty databases
2. `dumpdata.py:277` - Passes `object_count=0` to serializers.serialize()
3. `base.py:105` - Creates ProgressBar with `total_count=0`
4. `base.py:146` - Calls `progress_bar.update(count)`
5. `base.py:59` - Division by zero occurs

Related Django source files:
- `/django/core/serializers/base.py` (contains the buggy ProgressBar class)
- `/django/core/management/commands/dumpdata.py` (uses ProgressBar with potentially 0 object count)

## Proposed Fix

```diff
--- a/django/core/serializers/base.py
+++ b/django/core/serializers/base.py
@@ -56,6 +56,9 @@ class ProgressBar:
     def update(self, count):
         if not self.output:
             return
+        # Handle empty datasets gracefully
+        if self.total_count == 0:
+            return
         perc = count * 100 // self.total_count
         done = perc * self.progress_width // 100
         if self.prev_done >= done:
```