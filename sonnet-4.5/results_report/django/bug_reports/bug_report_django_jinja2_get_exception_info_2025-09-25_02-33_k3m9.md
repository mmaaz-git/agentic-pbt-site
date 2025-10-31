# Bug Report: django.template.backends.jinja2.get_exception_info IndexError

**Target**: `django.template.backends.jinja2.get_exception_info`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `get_exception_info` function crashes with an `IndexError` when the exception's `lineno` attribute is greater than the actual number of lines in the source code.

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
```

**Failing input**: `lineno=10, source="line 1\nline 2\nline 3"` (any case where `lineno > number of lines`)

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
```

## Why This Is A Bug

When a Jinja2 template exception reports a line number that exceeds the actual lines in the source (which can happen with malformed templates or corrupted exception state), the function crashes instead of gracefully handling the invalid line number. This prevents Django from displaying a proper debug page.

The bug is on line 106 of `django/template/backends/jinja2.py`:

```python
during = lines[lineno - 1][1]
```

This accesses the list without bounds checking. If `lineno > len(lines)`, it raises `IndexError`.

## Fix

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