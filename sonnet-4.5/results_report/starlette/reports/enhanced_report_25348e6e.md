# Bug Report: starlette.middleware.errors.ServerErrorMiddleware Generates Invalid HTML with Incomplete &nbsp Entities

**Target**: `starlette.middleware.errors.ServerErrorMiddleware.format_line`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `ServerErrorMiddleware.format_line` method generates invalid HTML by using incomplete HTML entities (`&nbsp` instead of `&nbsp;`), violating HTML5 specifications which require named character references to end with a semicolon.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from starlette.middleware.errors import ServerErrorMiddleware
import html


@given(st.text(min_size=1, max_size=100).filter(lambda x: ' ' in x))
def test_html_entity_completeness(line_with_spaces):
    middleware = ServerErrorMiddleware(None, debug=True)

    escaped = html.escape(line_with_spaces).replace(" ", "&nbsp")

    if "&nbsp" in escaped and "&nbsp;" not in escaped:
        raise AssertionError(
            f"Incomplete HTML entity found. "
            f"Missing semicolon in &nbsp entity. "
            f"Input: {repr(line_with_spaces)}, "
            f"Output: {escaped}"
        )


if __name__ == "__main__":
    # Run the property test
    test_html_entity_completeness()
```

<details>

<summary>
**Failing input**: `' '`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/42/hypo.py", line 23, in <module>
    test_html_entity_completeness()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/42/hypo.py", line 7, in test_html_entity_completeness
    def test_html_entity_completeness(line_with_spaces):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/42/hypo.py", line 13, in test_html_entity_completeness
    raise AssertionError(
    ...<4 lines>...
    )
AssertionError: Incomplete HTML entity found. Missing semicolon in &nbsp entity. Input: ' ', Output: &nbsp
Falsifying example: test_html_entity_completeness(
    line_with_spaces=' ',
)
```
</details>

## Reproducing the Bug

```python
import html
from starlette.middleware.errors import ServerErrorMiddleware

# Create a middleware instance with debug enabled
middleware = ServerErrorMiddleware(None, debug=True)

# Test line with leading spaces that should be converted to HTML entities
test_line = "    def example_function():"

# This is what the format_line method does internally (line 191 of errors.py)
escaped = html.escape(test_line).replace(" ", "&nbsp")

print("Input line:")
print(repr(test_line))
print("\nOutput from the buggy code:")
print(escaped)
print("\nExpected output (with proper HTML entities):")
print(html.escape(test_line).replace(" ", "&nbsp;"))
print("\nNote: The output is missing semicolons after '&nbsp' entities")
print("This violates HTML5 specification which requires named character references to end with semicolon")
```

<details>

<summary>
Invalid HTML entities generated with missing semicolons
</summary>
```
Input line:
'    def example_function():'

Output from the buggy code:
&nbsp&nbsp&nbsp&nbspdef&nbspexample_function():

Expected output (with proper HTML entities):
&nbsp;&nbsp;&nbsp;&nbsp;def&nbsp;example_function():

Note: The output is missing semicolons after '&nbsp' entities
This violates HTML5 specification which requires named character references to end with semicolon
```
</details>

## Why This Is A Bug

This violates the HTML5 specification which explicitly states that named character references must consist of an ampersand (&), followed by a name, followed by a semicolon (;). The code at line 191 of `starlette/middleware/errors.py` replaces spaces with `&nbsp` (missing the required semicolon) instead of the correct `&nbsp;`.

While most modern browsers implement error recovery and will render `&nbsp` as a non-breaking space despite the missing semicolon, this is classified as a "parse error" in HTML5. The invalid HTML can cause issues with:

1. HTML validators that check for standards compliance
2. Screen readers and accessibility tools that expect valid HTML
3. Strict HTML parsers that don't implement error recovery
4. XML parsers when used in XHTML mode
5. Potential future browser versions with stricter parsing rules

The bug only affects debug mode output (when `debug=True`), not production error responses.

## Relevant Context

The bug is located in the `format_line` method at `/home/npc/pbt/agentic-pbt/envs/starlette_env/lib/python3.13/site-packages/starlette/middleware/errors.py:191`. This method is responsible for formatting source code lines in debug traceback HTML output.

The method already performs HTML escaping for security (to handle `<` and `>` characters), showing that HTML validity is a concern for this code. The output is served with content-type "text/html" (line 254), confirming it's intended to be valid HTML.

HTML5 specification reference: https://html.spec.whatwg.org/multipage/syntax.html#named-character-references

Starlette source code: https://github.com/encode/starlette/blob/master/starlette/middleware/errors.py

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