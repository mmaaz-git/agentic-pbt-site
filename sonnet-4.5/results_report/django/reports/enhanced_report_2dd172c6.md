# Bug Report: django.template.backends.jinja2.get_exception_info Line Number Mismatch

**Target**: `django.template.backends.jinja2.get_exception_info`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `get_exception_info` function strips leading/trailing whitespace from template source before indexing lines, causing IndexErrors and incorrect line display when Jinja2 reports errors in templates with leading whitespace.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from hypothesis import given, strategies as st, assume, settings, example
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
    num_leading_newlines=st.integers(min_value=0, max_value=10),
)
def test_get_exception_info_line_indexing(lineno, num_lines, num_leading_newlines):
    assume(lineno <= num_lines + num_leading_newlines)
    assume(lineno > num_leading_newlines)  # The error must be in actual content, not empty lines

    # Create source with leading newlines
    leading = "\n" * num_leading_newlines
    lines_list = [f"line {i}" for i in range(1, num_lines + 1)]
    source = leading + "\n".join(lines_list)

    exc = MockException(
        filename="test.html",
        lineno=lineno,
        source=source
    )

    try:
        info = get_exception_info(exc)

        assert info["line"] == lineno
        # The expected line content considering the leading newlines
        if lineno <= num_leading_newlines:
            expected_during = ""  # Empty line
        else:
            expected_during = f"line {lineno - num_leading_newlines}"

        assert info["during"] == expected_during, (
            f"Line {lineno} should contain '{expected_during}', "
            f"but got '{info['during']}'"
        )
    except IndexError as e:
        # This is also a bug - it shouldn't raise IndexError
        print(f"IndexError for lineno={lineno}, num_lines={num_lines}, num_leading_newlines={num_leading_newlines}")
        print(f"Source has {len(source.split('\n'))} lines originally")
        print(f"After strip(), source has {len(source.strip().split('\n'))} lines")
        raise


if __name__ == "__main__":
    test_get_exception_info_line_indexing()
```

<details>

<summary>
**Failing input**: `test_get_exception_info_line_indexing(lineno=2, num_lines=1, num_leading_newlines=1)`
</summary>
```
IndexError for lineno=33, num_lines=30, num_leading_newlines=7
Source has 37 lines originally
After strip(), source has 30 lines
IndexError for lineno=33, num_lines=30, num_leading_newlines=7
Source has 37 lines originally
After strip(), source has 30 lines
IndexError for lineno=33, num_lines=30, num_leading_newlines=3
Source has 33 lines originally
After strip(), source has 30 lines
IndexError for lineno=31, num_lines=30, num_leading_newlines=3
Source has 33 lines originally
After strip(), source has 30 lines
IndexError for lineno=31, num_lines=30, num_leading_newlines=1
Source has 31 lines originally
After strip(), source has 30 lines
IndexError for lineno=31, num_lines=29, num_leading_newlines=2
Source has 31 lines originally
After strip(), source has 29 lines
IndexError for lineno=31, num_lines=28, num_leading_newlines=3
Source has 31 lines originally
After strip(), source has 28 lines
IndexError for lineno=31, num_lines=27, num_leading_newlines=4
Source has 31 lines originally
After strip(), source has 27 lines
IndexError for lineno=31, num_lines=26, num_leading_newlines=5
Source has 31 lines originally
After strip(), source has 26 lines
IndexError for lineno=31, num_lines=25, num_leading_newlines=6
Source has 31 lines originally
After strip(), source has 25 lines
IndexError for lineno=31, num_lines=23, num_leading_newlines=8
Source has 31 lines originally
After strip(), source has 23 lines
IndexError for lineno=31, num_lines=22, num_leading_newlines=9
Source has 31 lines originally
After strip(), source has 22 lines
IndexError for lineno=31, num_lines=21, num_leading_newlines=10
Source has 31 lines originally
After strip(), source has 21 lines
IndexError for lineno=30, num_lines=22, num_leading_newlines=10
Source has 32 lines originally
After strip(), source has 22 lines
IndexError for lineno=29, num_lines=23, num_leading_newlines=10
Source has 33 lines originally
After strip(), source has 23 lines
IndexError for lineno=28, num_lines=24, num_leading_newlines=10
Source has 34 lines originally
After strip(), source has 24 lines
IndexError for lineno=27, num_lines=25, num_leading_newlines=10
Source has 35 lines originally
After strip(), source has 25 lines
IndexError for lineno=27, num_lines=24, num_leading_newlines=9
Source has 33 lines originally
After strip(), source has 24 lines
IndexError for lineno=27, num_lines=23, num_leading_newlines=8
Source has 31 lines originally
After strip(), source has 23 lines
IndexError for lineno=27, num_lines=22, num_leading_newlines=7
Source has 29 lines originally
After strip(), source has 22 lines
IndexError for lineno=27, num_lines=21, num_leading_newlines=6
Source has 27 lines originally
After strip(), source has 21 lines
IndexError for lineno=26, num_lines=20, num_leading_newlines=6
Source has 26 lines originally
After strip(), source has 20 lines
IndexError for lineno=25, num_lines=19, num_leading_newlines=6
Source has 25 lines originally
After strip(), source has 19 lines
IndexError for lineno=24, num_lines=18, num_leading_newlines=6
Source has 24 lines originally
After strip(), source has 18 lines
IndexError for lineno=23, num_lines=17, num_leading_newlines=6
Source has 23 lines originally
After strip(), source has 17 lines
IndexError for lineno=22, num_lines=16, num_leading_newlines=6
Source has 22 lines originally
After strip(), source has 16 lines
IndexError for lineno=17, num_lines=11, num_leading_newlines=6
Source has 17 lines originally
After strip(), source has 11 lines
IndexError for lineno=7, num_lines=1, num_leading_newlines=6
Source has 7 lines originally
After strip(), source has 1 lines
IndexError for lineno=6, num_lines=1, num_leading_newlines=5
Source has 6 lines originally
After strip(), source has 1 lines
IndexError for lineno=5, num_lines=1, num_leading_newlines=4
Source has 5 lines originally
After strip(), source has 1 lines
IndexError for lineno=4, num_lines=1, num_leading_newlines=3
Source has 4 lines originally
After strip(), source has 1 lines
IndexError for lineno=3, num_lines=1, num_leading_newlines=2
Source has 3 lines originally
After strip(), source has 1 lines
IndexError for lineno=2, num_lines=1, num_leading_newlines=1
Source has 2 lines originally
After strip(), source has 1 lines
IndexError for lineno=2, num_lines=1, num_leading_newlines=1
Source has 2 lines originally
After strip(), source has 1 lines
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/9/hypo.py", line 60, in <module>
  |     test_get_exception_info_line_indexing()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/9/hypo.py", line 17, in test_get_exception_info_line_indexing
  |     @given(
  |
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/9/hypo.py", line 47, in test_get_exception_info_line_indexing
    |     assert info["during"] == expected_during, (
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    | AssertionError: Line 2 should contain 'line 1', but got 'line 2'
    | Falsifying example: test_get_exception_info_line_indexing(
    |     lineno=2,
    |     num_lines=2,  # or any other generated value
    |     num_leading_newlines=1,
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/pbt/agentic-pbt/worker_/9/hypo.py:48
    |         /home/npc/pbt/agentic-pbt/worker_/9/hypo.py:51
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/9/hypo.py", line 38, in test_get_exception_info_line_indexing
    |     info = get_exception_info(exc)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/django/template/backends/jinja2.py", line 106, in get_exception_info
    |     during = lines[lineno - 1][1]
    |              ~~~~~^^^^^^^^^^^^
    | IndexError: list index out of range
    | Falsifying example: test_get_exception_info_line_indexing(
    |     lineno=2,
    |     num_lines=1,
    |     num_leading_newlines=1,
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/pbt/agentic-pbt/worker_/9/hypo.py:51
    +------------------------------------
```
</details>

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


# Test case 1: Source with leading newlines - this will cause IndexError
print("Test 1: Source with leading newlines (will cause IndexError)")
print("=" * 60)

source_with_leading_newlines = "\n\nline 3 content\nline 4 content"

exc = MockJinjaException(
    filename="template.html",
    lineno=3,
    source=source_with_leading_newlines,
    message="syntax error"
)

print(f"Original source (repr): {repr(source_with_leading_newlines)}")
print(f"Original source has {len(source_with_leading_newlines.split('\n'))} lines")
print(f"After strip(), source has {len(source_with_leading_newlines.strip().split('\n'))} lines")
print(f"Jinja2 reported error at line: {exc.lineno}")
print(f"Expected content at line 3: 'line 3 content'")

try:
    info = get_exception_info(exc)
    print(f"Actual content returned: {repr(info['during'])}")
except IndexError as e:
    print(f"ERROR: IndexError occurred: {e}")
    print("This happens because after stripping, there are only 2 lines, but we're trying to access line 3")

print()

# Test case 2: Source with more lines - wrong line displayed
print("Test 2: Source with more lines (wrong line displayed)")
print("=" * 60)

source_with_more_lines = "\n\nActual line 3\nActual line 4\nActual line 5"

exc2 = MockJinjaException(
    filename="template2.html",
    lineno=3,
    source=source_with_more_lines,
    message="syntax error"
)

print(f"Original source (repr): {repr(source_with_more_lines)}")
print(f"Original source lines: {source_with_more_lines.split('\n')}")
print(f"Jinja2 reported error at line: {exc2.lineno}")
print(f"Expected content at line 3: 'Actual line 3'")

info2 = get_exception_info(exc2)
print(f"Actual content returned: {repr(info2['during'])}")
print(f"This is wrong! It's showing line 3 of the stripped source, which is 'Actual line 5'")
```

<details>

<summary>
IndexError and wrong line display when templates have leading newlines
</summary>
```
Test 1: Source with leading newlines (will cause IndexError)
============================================================
Original source (repr): '\n\nline 3 content\nline 4 content'
Original source has 4 lines
After strip(), source has 2 lines
Jinja2 reported error at line: 3
Expected content at line 3: 'line 3 content'
ERROR: IndexError occurred: list index out of range
This happens because after stripping, there are only 2 lines, but we're trying to access line 3

Test 2: Source with more lines (wrong line displayed)
============================================================
Original source (repr): '\n\nActual line 3\nActual line 4\nActual line 5'
Original source lines: ['', '', 'Actual line 3', 'Actual line 4', 'Actual line 5']
Jinja2 reported error at line: 3
Expected content at line 3: 'Actual line 3'
Actual content returned: 'Actual line 5'
This is wrong! It's showing line 3 of the stripped source, which is 'Actual line 5'
```
</details>

## Why This Is A Bug

This violates the expected behavior of error reporting in Django's Jinja2 template backend. When Jinja2 reports an error at line N, developers expect `get_exception_info` to display the content from line N of their original template source. However, the function strips leading/trailing whitespace before indexing lines (line 105 in `/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/template/backends/jinja2.py`), causing two critical issues:

1. **IndexError crashes**: When the stripped source has fewer lines than the original, accessing the reported line number causes an IndexError, crashing the debug page itself.

2. **Incorrect line display**: When the line exists but at a different position after stripping, the wrong line content is shown, misleading developers about where the error occurred.

The function's docstring states it should "Format exception information for display on the debug page", but showing the wrong line or crashing defeats this debugging purpose. Jinja2's line numbers always refer to the original source, so Django must preserve that line numbering to accurately display errors.

## Relevant Context

This bug affects Django's template debugging functionality, which is crucial for development workflows. Templates with leading whitespace are common in real applications:
- Templates with heredocs or multiline strings
- Generated templates with formatting whitespace
- Templates included from other files that may have extra newlines
- Template inheritance where child templates may have leading newlines

The issue exists in the current Django release and affects the Jinja2 template backend specifically. The Django template backend doesn't have this issue because it handles line numbering differently.

Code location: `/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/template/backends/jinja2.py:105`

Django documentation on template debugging: https://docs.djangoproject.com/en/stable/ref/templates/api/#debug

## Proposed Fix

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