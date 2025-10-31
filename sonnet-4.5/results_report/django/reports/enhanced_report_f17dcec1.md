# Bug Report: django.template.backends.jinja2.get_exception_info IndexError on Out-of-Bounds Line Numbers

**Target**: `django.template.backends.jinja2.get_exception_info`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `get_exception_info` function crashes with an `IndexError` when processing Jinja2 template exceptions that report a line number greater than the actual number of lines in the source code, preventing Django from displaying debugging information.

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


if __name__ == "__main__":
    test_get_exception_info_no_crash()
```

<details>

<summary>
**Failing input**: `source='', lineno=2, message=''`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/31/hypo.py", line 31, in <module>
    test_get_exception_info_no_crash()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/31/hypo.py", line 12, in test_get_exception_info_no_crash
    @given(

  File "/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/31/hypo.py", line 26, in test_get_exception_info_no_crash
    info = get_exception_info(exc)
  File "/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/template/backends/jinja2.py", line 106, in get_exception_info
    during = lines[lineno - 1][1]
             ~~~~~^^^^^^^^^^^^
IndexError: list index out of range
Falsifying example: test_get_exception_info_no_crash(
    # The test always failed when commented parts were varied together.
    source='',  # or any other generated value
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
print(info)
```

<details>

<summary>
IndexError at line 106 when accessing out-of-bounds list index
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/31/repo.py", line 25, in <module>
    info = get_exception_info(exc)
  File "/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/template/backends/jinja2.py", line 106, in get_exception_info
    during = lines[lineno - 1][1]
             ~~~~~^^^^^^^^^^^^
IndexError: list index out of range
```
</details>

## Why This Is A Bug

This violates the expected behavior of a debugging utility function. The `get_exception_info` function is documented as "Format exception information for display on the debug page" and is specifically called when Django catches Jinja2 TemplateSyntaxError exceptions (lines 46-47 and 76-77 in jinja2.py). When this function crashes, Django cannot display the debug page at all, completely defeating its purpose of helping developers debug template errors.

The function already handles the case where `source` is None (lines 100-103 and 110-113), showing it's designed to be resilient to missing data. However, it fails to validate that `lineno` is within bounds before accessing `lines[lineno - 1]` at line 106. This can occur in real-world scenarios when:
- Jinja2 has a bug that reports incorrect line numbers
- Template compilation generates invalid exception state
- Templates are dynamically modified after initial parsing
- Testing frameworks use mock exceptions with synthetic data

## Relevant Context

The bug occurs in Django's Jinja2 backend, which integrates the Jinja2 template engine with Django's template system. The `get_exception_info` function is critical for Django's development experience as it formats template syntax errors for the debug error page.

Key code locations:
- Bug location: `/django/template/backends/jinja2.py:106`
- Function definition: `/django/template/backends/jinja2.py:92-125`
- Called from: Lines 47 and 77 when catching `jinja2.TemplateSyntaxError`

The function correctly handles missing source (when `source is None`) but doesn't handle invalid line numbers. The Jinja2 documentation doesn't guarantee that `lineno` will always be within valid bounds, and the function should be defensive against such cases.

## Proposed Fix

```diff
--- a/django/template/backends/jinja2.py
+++ b/django/template/backends/jinja2.py
@@ -103,7 +103,10 @@ def get_exception_info(exception):
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