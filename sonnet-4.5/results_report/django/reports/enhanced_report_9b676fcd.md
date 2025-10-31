# Bug Report: django.template.backends.jinja2.get_exception_info IndexError and Incorrect Line Count for Empty Templates

**Target**: `django.template.backends.jinja2.get_exception_info`
**Severity**: Medium
**Bug Type**: Crash, Logic
**Date**: 2025-09-25

## Summary

The `get_exception_info` function crashes with IndexError when processing empty templates with `lineno > 1`, and incorrectly reports 1 line for empty templates instead of 0.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

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

    assert info['total'] == expected_total, f"Expected {expected_total} lines but got {info['total']} for source: {repr(source)}"

# Run the test
if __name__ == "__main__":
    test_get_exception_info_total_lines()
```

<details>

<summary>
**Failing input**: `lineno=2, source='', message=''` and `lineno=1, source='', message=''`
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/54/hypo.py", line 38, in <module>
  |     test_get_exception_info_total_lines()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/54/hypo.py", line 15, in test_get_exception_info_total_lines
  |     st.integers(min_value=1, max_value=10000),
  |                ^^^
  |   File "/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/54/hypo.py", line 27, in test_get_exception_info_total_lines
    |     info = get_exception_info(exc)
    |   File "/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/template/backends/jinja2.py", line 106, in get_exception_info
    |     during = lines[lineno - 1][1]
    |              ~~~~~^^^^^^^^^^^^
    | IndexError: list index out of range
    | Falsifying example: test_get_exception_info_total_lines(
    |     lineno=2,
    |     source='',
    |     message='',
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/54/hypo.py", line 34, in test_get_exception_info_total_lines
    |     assert info['total'] == expected_total, f"Expected {expected_total} lines but got {info['total']} for source: {repr(source)}"
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    | AssertionError: Expected 0 lines but got 1 for source: ''
    | Falsifying example: test_get_exception_info_total_lines(
    |     lineno=1,
    |     source='',
    |     message='',
    | )
    +------------------------------------
```
</details>

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

# Test 1: Empty source (original bug report)
print("Test 1: Empty source")
print("-" * 40)
exc = MockJinja2Exception(lineno=1, source='', message='test error', filename='test.html')
info = get_exception_info(exc)
print(f"Empty source has {info['total']} lines (expected 0)")
print(f"source_lines: {info['source_lines']}")
print(f"during: '{info['during']}'")
print()

# Test 2: Empty source with lineno > 1 (this should crash)
print("Test 2: Empty source with lineno=2")
print("-" * 40)
try:
    exc2 = MockJinja2Exception(lineno=2, source='', message='test error', filename='test.html')
    info2 = get_exception_info(exc2)
    print(f"Result: total={info2['total']}")
except IndexError as e:
    print(f"ERROR: IndexError occurred: {e}")
    import traceback
    traceback.print_exc()
print()

# Test 3: Whitespace-only source
print("Test 3: Whitespace-only source")
print("-" * 40)
exc3 = MockJinja2Exception(lineno=1, source='   \n  \t  ', message='test error', filename='test.html')
info3 = get_exception_info(exc3)
print(f"Whitespace-only source has {info3['total']} lines (expected 0)")
print(f"source_lines: {info3['source_lines']}")
print(f"during: '{info3['during']}'")
```

<details>

<summary>
IndexError crash when lineno > 1 on empty source
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/54/repo.py", line 28, in <module>
    info2 = get_exception_info(exc2)
  File "/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/template/backends/jinja2.py", line 106, in get_exception_info
    during = lines[lineno - 1][1]
             ~~~~~^^^^^^^^^^^^
IndexError: list index out of range
Test 1: Empty source
----------------------------------------
Empty source has 1 lines (expected 0)
source_lines: [(1, '')]
during: ''

Test 2: Empty source with lineno=2
----------------------------------------
ERROR: IndexError occurred: list index out of range

Test 3: Whitespace-only source
----------------------------------------
Whitespace-only source has 1 lines (expected 0)
source_lines: [(1, '')]
during: ''
```
</details>

## Why This Is A Bug

This violates expected behavior in two critical ways:

1. **IndexError Crash**: When Jinja2 reports an error on line 2+ of an empty template (which can happen with missing template files or dynamically generated empty templates), Django's error handling code itself crashes with an IndexError. This breaks the debug page entirely instead of displaying helpful error information. The code at line 106 (`during = lines[lineno - 1][1]`) attempts to access an index that doesn't exist when the template is empty but lineno > 1.

2. **Incorrect Line Count**: Empty templates report having 1 line instead of 0. This occurs because Python's `''.split('\n')` returns `['']` (a list with one empty string) rather than an empty list. This contradicts the intuitive expectation that an empty file has zero lines, not one empty line.

The function's docstring states it formats exception information "for display on the debug page," meaning it's part of Django's error handling infrastructure. Error handling code should be robust and never crash, especially on edge cases like empty templates.

## Relevant Context

The bug occurs in `/django/template/backends/jinja2.py` at lines 105-106:
```python
lines = list(enumerate(source.strip().split("\n"), start=1))
during = lines[lineno - 1][1]
```

This function is called from two places in the same file:
- Line 47: When catching `jinja2.TemplateSyntaxError` in `get_template()`
- Line 77: When catching `jinja2.TemplateSyntaxError` in `Template.render()`

Both call sites wrap Jinja2 exceptions to add debug information via `get_exception_info()`. If this function crashes, the entire error reporting mechanism fails.

Empty templates can occur in several scenarios:
- Template files that exist but are empty during development
- Dynamically generated templates that evaluate to empty strings
- Template files that become empty due to version control conflicts or file system issues

Django's debug page documentation: https://docs.djangoproject.com/en/stable/ref/templates/api/#django.template.Template.render

## Proposed Fix

```diff
--- a/django/template/backends/jinja2.py
+++ b/django/template/backends/jinja2.py
@@ -102,10 +102,17 @@ def get_exception_info(exception):
         if exception_file.exists():
             source = exception_file.read_text()
     if source is not None:
-        lines = list(enumerate(source.strip().split("\n"), start=1))
-        during = lines[lineno - 1][1]
+        stripped = source.strip()
+        if stripped:
+            lines = list(enumerate(stripped.split("\n"), start=1))
+        else:
+            lines = []
+
         total = len(lines)
-        top = max(0, lineno - context_lines - 1)
+        if lineno <= total and lineno > 0:
+            during = lines[lineno - 1][1]
+        else:
+            during = ""
+        top = max(0, min(lineno - context_lines - 1, total))
         bottom = min(total, lineno + context_lines)
     else:
         during = ""
```