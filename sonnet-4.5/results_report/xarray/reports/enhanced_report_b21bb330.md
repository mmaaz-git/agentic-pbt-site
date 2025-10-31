# Bug Report: django.template.backends.jinja2.get_exception_info IndexError on Invalid Line Numbers

**Target**: `django.template.backends.jinja2.get_exception_info`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `get_exception_info` function crashes with an `IndexError` when processing Jinja2 template exceptions that report line numbers exceeding the actual number of lines in the source code, preventing Django from displaying proper debug pages.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'django.conf.global_settings'

from hypothesis import given, strategies as st, settings
from django.template.backends.jinja2 import get_exception_info


@settings(max_examples=500)
@given(
    st.text(min_size=0, max_size=100),
    st.integers(min_value=1, max_value=200),
    st.text(max_size=50)
)
def test_get_exception_info_no_crash(source, lineno, message):
    class MockException:
        def __init__(self):
            self.lineno = lineno
            self.source = source
            self.filename = 'test.html'
            self.message = message

    exc = MockException()
    info = get_exception_info(exc)
    assert info['line'] == lineno

if __name__ == '__main__':
    test_get_exception_info_no_crash()
```

<details>

<summary>
**Failing input**: `source='', lineno=2, message=''`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/62/hypo.py", line 30, in <module>
    test_get_exception_info_no_crash()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/62/hypo.py", line 12, in test_get_exception_info_no_crash
    @given(

  File "/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/62/hypo.py", line 26, in test_get_exception_info_no_crash
    info = get_exception_info(exc)
  File "/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/template/backends/jinja2.py", line 106, in get_exception_info
    during = lines[lineno - 1][1]
             ~~~~~^^^^^^^^^^^^
IndexError: list index out of range
Falsifying example: test_get_exception_info_no_crash(
    source='',
    lineno=2,
    message='',  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'django.conf.global_settings'

from django.template.backends.jinja2 import get_exception_info


class MockException:
    def __init__(self, lineno, source, filename, message):
        self.lineno = lineno
        self.source = source
        self.filename = filename
        self.message = message


exc = MockException(
    lineno=10,
    source="line 1\nline 2\nline 3",
    filename="test.html",
    message="test error"
)

info = get_exception_info(exc)
print("Success! Info:", info)
```

<details>

<summary>
IndexError when lineno exceeds source lines
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/62/repo.py", line 25, in <module>
    info = get_exception_info(exc)
  File "/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/template/backends/jinja2.py", line 106, in get_exception_info
    during = lines[lineno - 1][1]
             ~~~~~^^^^^^^^^^^^
IndexError: list index out of range
```
</details>

## Why This Is A Bug

This violates expected behavior for several critical reasons:

1. **Purpose Contradiction**: The function's documented purpose is to "Format exception information for display on the debug page." Crashing during debug information generation defeats this fundamental purpose.

2. **Defensive Programming Inconsistency**: The function already implements defensive programming for edge cases - it explicitly handles `source is None` (lines 100-103) by attempting to read from a file and setting safe defaults if that fails (lines 111-113). However, it fails to apply the same defensive approach to line number validation.

3. **Debug Page Failure**: When this crash occurs, Django cannot display its debug page at all. The crash happens in both `Jinja2.get_template()` (line 47) and `Template.render()` (line 77) when catching `jinja2.TemplateSyntaxError`, preventing users from seeing any debugging information about their template errors.

4. **Real-World Scenarios**: Invalid line numbers can occur legitimately through:
   - Malformed or corrupted templates
   - Template preprocessing that modifies line counts
   - Jinja2 internal errors or edge cases
   - Dynamic template generation with inconsistent metadata

5. **List Access Without Bounds Checking**: Line 106 directly accesses `lines[lineno - 1][1]` without verifying that the index exists, despite `lineno` being user-controllable through the exception object.

## Relevant Context

The bug occurs at line 106 in `/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/template/backends/jinja2.py`:

```python
105: lines = list(enumerate(source.strip().split("\n"), start=1))
106: during = lines[lineno - 1][1]  # <-- Crashes here when lineno > len(lines)
107: total = len(lines)
```

The function is called from two locations:
- Line 47: When catching `jinja2.TemplateSyntaxError` in `get_template()`
- Line 77: When catching `jinja2.TemplateSyntaxError` in `Template.render()`

Both callers expect a valid dictionary to be returned for the `template_debug` attribute, which is used by Django's debug middleware to display template errors to developers.

Django documentation on template debugging: https://docs.djangoproject.com/en/stable/ref/templates/api/#django.template.Template.template_debug

## Proposed Fix

```diff
--- a/django/template/backends/jinja2.py
+++ b/django/template/backends/jinja2.py
@@ -103,7 +103,11 @@ def get_exception_info(exception):
             source = exception_file.read_text()
     if source is not None:
         lines = list(enumerate(source.strip().split("\n"), start=1))
-        during = lines[lineno - 1][1]
+        if 0 <= lineno - 1 < len(lines):
+            during = lines[lineno - 1][1]
+        else:
+            during = ""
         total = len(lines)
         top = max(0, lineno - context_lines - 1)
         bottom = min(total, lineno + context_lines)
```