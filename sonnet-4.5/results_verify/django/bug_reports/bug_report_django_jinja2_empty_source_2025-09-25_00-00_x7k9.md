# Bug Report: django.template.backends.jinja2 Incorrect Line Count for Empty Source

**Target**: `django.template.backends.jinja2.get_exception_info`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `get_exception_info` function incorrectly reports that an empty template source has 1 line instead of 0 lines. This happens because `''.strip().split('\n')` returns `['']` (a list with one empty string) rather than an empty list.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.template.backends.jinja2 import get_exception_info

class MockJinja2Exception:
    def __init__(self, lineno, source, message, filename):
        self.lineno = lineno
        self.source = source
        self.message = message
        self.filename = filename

@given(
    st.integers(min_value=1, max_value=10000),
    st.text(min_size=0, max_size=1000),
    st.text(max_size=100)
)
def test_get_exception_info_total_lines(lineno, source, message):
    exc = MockJinja2Exception(
        lineno=lineno,
        source=source,
        message=message,
        filename="test.html"
    )

    info = get_exception_info(exc)

    if source.strip():
        expected_total = len(source.strip().split('\n'))
    else:
        expected_total = 0

    assert info['total'] == expected_total
```

**Failing input**: `lineno=1, source='', message='test'`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.template.backends.jinja2 import get_exception_info

class MockJinja2Exception:
    def __init__(self, lineno, source, message, filename):
        self.lineno = lineno
        self.source = source
        self.message = message
        self.filename = filename

exc = MockJinja2Exception(lineno=1, source='', message='test', filename='test.html')
info = get_exception_info(exc)

print(f"Empty source has {info['total']} lines (expected 0)")
```

Output:
```
Empty source has 1 lines (expected 0)
```

## Why This Is A Bug

The debug information incorrectly reports that an empty template has 1 line. This is because Python's `str.split('\n')` on an empty string returns a list with one empty element `['']` rather than an empty list `[]`.

While empty templates are rare, the debug information should accurately reflect the source. This could confuse developers debugging template issues.

## Fix

```diff
--- a/django/template/backends/jinja2.py
+++ b/django/template/backends/jinja2.py
@@ -102,7 +102,10 @@ def get_exception_info(exception):
             source = exception_file.read_text()
     if source is not None:
-        lines = list(enumerate(source.strip().split("\n"), start=1))
+        stripped = source.strip()
+        if stripped:
+            lines = list(enumerate(stripped.split("\n"), start=1))
+        else:
+            lines = []
         during = lines[lineno - 1][1]
         total = len(lines)
         top = max(0, lineno - context_lines - 1)
```