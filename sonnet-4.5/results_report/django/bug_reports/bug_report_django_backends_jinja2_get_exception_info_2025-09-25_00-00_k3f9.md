# Bug Report: django.template.backends.jinja2.get_exception_info Line Number Mismatch

**Target**: `django.template.backends.jinja2.get_exception_info`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `get_exception_info` function incorrectly strips leading/trailing whitespace from template source before indexing lines, causing a mismatch between Jinja2's reported line numbers and the actual line content displayed in error messages.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from hypothesis import given, strategies as st, assume, settings
from django.template.backends.jinja2 import get_exception_info


class MockException:
    def __init__(self, filename, lineno, source):
        self.filename = filename
        self.lineno = lineno
        self.source = source
        self.message = "Test error"


@settings(max_examples=500)
@given(
    lineno=st.integers(min_value=1, max_value=100),
    num_lines=st.integers(min_value=1, max_value=100),
)
def test_get_exception_info_line_indexing(lineno, num_lines):
    assume(lineno <= num_lines)

    lines_list = [f"line {i}" for i in range(1, num_lines + 1)]
    source = "\n".join(lines_list)

    exc = MockException(
        filename="test.html",
        lineno=lineno,
        source=source
    )

    info = get_exception_info(exc)

    assert info["line"] == lineno
    expected_during = f"line {lineno}"
    assert info["during"] == expected_during, (
        f"Line {lineno} should contain '{expected_during}', "
        f"but got '{info['during']}'"
    )
```

**Failing input**: Template source with leading newlines, e.g., `"\n\nline 3 content\nline 4 content"` with `lineno=3`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.template.backends.jinja2 import get_exception_info


class MockJinjaException:
    def __init__(self, filename, lineno, source, message):
        self.filename = filename
        self.lineno = lineno
        self.source = source
        self.message = message


source_with_leading_newlines = "\n\nline 3 content\nline 4 content"

exc = MockJinjaException(
    filename="template.html",
    lineno=3,
    source=source_with_leading_newlines,
    message="syntax error"
)

info = get_exception_info(exc)

print(f"Jinja2 reported line: {exc.lineno}")
print(f"Expected content: 'line 3 content'")
print(f"Actual content: {repr(info['during'])}")
```

## Why This Is A Bug

The function's docstring states it should "Format exception information for display on the debug page". When a Jinja2 template has leading whitespace (common in templates with heredocs or multiline strings), the line numbers reported by Jinja2 correspond to the original source. However, `get_exception_info` strips this whitespace before indexing, causing the wrong line to be displayed in the debug output. This misleads developers trying to debug template errors.

## Fix

```diff
--- a/django/template/backends/jinja2.py
+++ b/django/template/backends/jinja2.py
@@ -102,7 +102,7 @@ def get_exception_info(exception):
         if exception_file.exists():
             source = exception_file.read_text()
     if source is not None:
-        lines = list(enumerate(source.strip().split("\n"), start=1))
+        lines = list(enumerate(source.split("\n"), start=1))
         during = lines[lineno - 1][1]
         total = len(lines)
         top = max(0, lineno - context_lines - 1)
```