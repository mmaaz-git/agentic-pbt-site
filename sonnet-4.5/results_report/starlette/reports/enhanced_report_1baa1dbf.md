# Bug Report: ServerErrorMiddleware Generates Invalid HTML Entities

**Target**: `starlette.middleware.errors.ServerErrorMiddleware`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

ServerErrorMiddleware generates invalid HTML by using `&nbsp` without a semicolon instead of the correct HTML entity `&nbsp;` when formatting code lines in debug mode error pages.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, HealthCheck
import html


@given(st.text(min_size=1, max_size=100).map(lambda x: " " + x if " " not in x else x))
@settings(suppress_health_check=[HealthCheck.filter_too_much])
def test_nbsp_entity_correctness(line):
    # Only test if line contains spaces
    if " " not in line:
        return

    # This is what Starlette does (line 191 in errors.py)
    result = html.escape(line).replace(" ", "&nbsp")

    # Check if the HTML entity is correct (should have semicolon)
    assert "&nbsp;" in result, (
        f"HTML entity for non-breaking space should be '&nbsp;' with semicolon. "
        f"Got '{result}' which contains '&nbsp' without semicolon"
    )


if __name__ == "__main__":
    # Run the property-based test
    test_nbsp_entity_correctness()
```

<details>

<summary>
**Failing input**: `line=' 0'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/31/hypo.py", line 24, in <module>
    test_nbsp_entity_correctness()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/31/hypo.py", line 6, in test_nbsp_entity_correctness
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/31/hypo.py", line 16, in test_nbsp_entity_correctness
    assert "&nbsp;" in result, (
           ^^^^^^^^^^^^^^^^^^
AssertionError: HTML entity for non-breaking space should be '&nbsp;' with semicolon. Got '&nbsp0' which contains '&nbsp' without semicolon
Falsifying example: test_nbsp_entity_correctness(
    line=' 0',
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/31/hypo.py:17
```
</details>

## Reproducing the Bug

```python
import html

# Demonstrating the bug in Starlette's ServerErrorMiddleware
# This shows how spaces are replaced with invalid HTML entities

def demonstrate_bug():
    # Example line of code that might appear in a traceback
    line = "    def example_function():"

    # This is what Starlette currently does (line 191 in errors.py)
    buggy_result = html.escape(line).replace(" ", "&nbsp")

    # This is what it should do
    correct_result = html.escape(line).replace(" ", "&nbsp;")

    print("Original line:")
    print(f"'{line}'")
    print()

    print("Buggy result (missing semicolons):")
    print(f"'{buggy_result}'")
    print()

    print("Correct result (with semicolons):")
    print(f"'{correct_result}'")
    print()

    # Demonstrate the issue with another example
    print("=" * 50)
    print("Another example with multiple spaces:")
    line2 = "        return x + y  # Add two numbers"
    buggy_result2 = html.escape(line2).replace(" ", "&nbsp")
    correct_result2 = html.escape(line2).replace(" ", "&nbsp;")

    print(f"Original: '{line2}'")
    print(f"Buggy:    '{buggy_result2}'")
    print(f"Correct:  '{correct_result2}'")
    print()

    # Show the difference
    print("The difference:")
    print(f"Invalid entity: '&nbsp' (no semicolon)")
    print(f"Valid entity:   '&nbsp;' (with semicolon)")
    print()
    print("According to HTML spec, named character references MUST end with semicolon.")

if __name__ == "__main__":
    demonstrate_bug()
```

<details>

<summary>
Invalid HTML entities generated for spaces in code
</summary>
```
Original line:
'    def example_function():'

Buggy result (missing semicolons):
'&nbsp&nbsp&nbsp&nbspdef&nbspexample_function():'

Correct result (with semicolons):
'&nbsp;&nbsp;&nbsp;&nbsp;def&nbsp;example_function():'

==================================================
Another example with multiple spaces:
Original: '        return x + y  # Add two numbers'
Buggy:    '&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbspreturn&nbspx&nbsp+&nbspy&nbsp&nbsp#&nbspAdd&nbsptwo&nbspnumbers'
Correct:  '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return&nbsp;x&nbsp;+&nbsp;y&nbsp;&nbsp;#&nbsp;Add&nbsp;two&nbsp;numbers'

The difference:
Invalid entity: '&nbsp' (no semicolon)
Valid entity:   '&nbsp;' (with semicolon)

According to HTML spec, named character references MUST end with semicolon.
```
</details>

## Why This Is A Bug

The HTML specification (https://html.spec.whatwg.org/) explicitly requires that named character references must end with a semicolon. The entity `&nbsp` without a semicolon is not valid HTML.

In the file `/home/npc/pbt/agentic-pbt/envs/starlette_env/lib/python3.13/site-packages/starlette/middleware/errors.py`, line 191 contains:

```python
"line": html.escape(line).replace(" ", "&nbsp"),
```

This generates invalid HTML entities whenever spaces appear in code lines displayed in debug mode tracebacks. While modern browsers typically handle this through error recovery mechanisms, the HTML is technically invalid and violates web standards. HTML validators will flag this as an error, and some tools (particularly accessibility tools or strict parsers) may not handle the malformed entities correctly.

## Relevant Context

The bug affects the `format_line` method in `ServerErrorMiddleware` which formats code context lines for display in debug error pages. Every space character in the source code shown in tracebacks is replaced with the malformed entity `&nbsp` instead of the correct `&nbsp;`.

The HTML Living Standard states in section 13.5 (Named character references): "Named character references must be terminated by a U+003B SEMICOLON character (;)."

While the spec includes some legacy entities without semicolons for backward compatibility (like `&lt` for `<`), these are explicitly marked as parse errors and should not be used by content authors. The `&nbsp` entity specifically requires a semicolon for valid HTML.

Relevant documentation:
- HTML Living Standard on named character references: https://html.spec.whatwg.org/multipage/named-characters.html
- Starlette ServerErrorMiddleware source: https://github.com/encode/starlette/blob/master/starlette/middleware/errors.py

## Proposed Fix

```diff
--- a/starlette/middleware/errors.py
+++ b/starlette/middleware/errors.py
@@ -188,7 +188,7 @@ class ServerErrorMiddleware:
     def format_line(self, index: int, line: str, frame_lineno: int, frame_index: int) -> str:
         values = {
             # HTML escape - line could contain < or >
-            "line": html.escape(line).replace(" ", "&nbsp"),
+            "line": html.escape(line).replace(" ", "&nbsp;"),
             "lineno": (frame_lineno - frame_index) + index,
         }
```