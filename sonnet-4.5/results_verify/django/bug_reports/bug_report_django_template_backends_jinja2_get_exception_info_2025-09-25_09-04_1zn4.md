# Bug Report: django.template.backends.jinja2.get_exception_info IndexError on Invalid Line Numbers

**Target**: `django.template.backends.jinja2.get_exception_info`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `get_exception_info` function crashes with an `IndexError` when the exception's `lineno` attribute is out of bounds for the source code (either greater than the total lines or non-positive).

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from django.template.backends.jinja2 import get_exception_info


@given(
    lineno=st.integers(min_value=1),
    source=st.text(),
    filename=st.text(min_size=1),
    message=st.text(),
)
@settings(max_examples=500)
def test_get_exception_info_doesnt_crash(lineno, source, filename, message):
    class MockException:
        pass

    exc = MockException()
    exc.lineno = lineno
    exc.filename = filename
    exc.message = message
    exc.source = source

    result = get_exception_info(exc)

    assert isinstance(result, dict)
```

**Failing input**: `lineno=2, source='', filename='0', message=''`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'test_settings')

from django.template.backends.jinja2 import get_exception_info

class MockException:
    lineno = 2
    filename = 'template.html'
    message = 'Syntax error'
    source = ''

exc = MockException()
result = get_exception_info(exc)
```

## Why This Is A Bug

The `get_exception_info` function is designed to format exception information for display on Django's debug page. It should handle edge cases gracefully, but instead crashes when:
1. The reported line number exceeds the actual number of lines in the source
2. The line number is zero or negative

This violates the expected behavior of a debug helper function, which should not raise exceptions during error formatting. In production, this could occur if there's a mismatch between Jinja2's reported line numbers and the actual template content.

## Fix

```diff
--- a/django/template/backends/jinja2.py
+++ b/django/template/backends/jinja2.py
@@ -103,7 +103,10 @@ def get_exception_info(exception):
             source = exception_file.read_text()
     if source is not None:
         lines = list(enumerate(source.strip().split("\n"), start=1))
-        during = lines[lineno - 1][1]
+        if 0 < lineno <= len(lines):
+            during = lines[lineno - 1][1]
+        else:
+            during = ""
         total = len(lines)
         top = max(0, lineno - context_lines - 1)
         bottom = min(total, lineno + context_lines)
```