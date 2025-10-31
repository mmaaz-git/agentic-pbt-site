# Bug Report: ServerErrorMiddleware Incomplete HTML Entity

**Target**: `starlette.middleware.errors.ServerErrorMiddleware.format_line`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `ServerErrorMiddleware.format_line` method generates invalid HTML by using incomplete HTML entities (`&nbsp` instead of `&nbsp;`), violating HTML5 specifications.

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
            f"Missing semicolon in &nbsp entity"
        )
```

**Failing input**: any string with spaces, e.g., `"def foo():"`

## Reproducing the Bug

```python
import html
from starlette.middleware.errors import ServerErrorMiddleware

middleware = ServerErrorMiddleware(None, debug=True)
test_line = "    def example_function():"

escaped = html.escape(test_line).replace(" ", "&nbsp")

print(escaped)
```

Output: `&nbsp&nbsp&nbsp&nbspdef&nbspexample_function():`

Expected: `&nbsp;&nbsp;&nbsp;&nbsp;def&nbsp;example_function():`

## Why This Is A Bug

HTML5 specifications require named character references (entities) to end with a semicolon. The code at `starlette/middleware/errors.py:191` replaces spaces with `&nbsp` (missing semicolon) instead of `&nbsp;`.

While most modern browsers render `&nbsp` correctly due to error recovery, this violates the HTML5 standard and can cause issues with:
- Strict HTML validators
- Screen readers and accessibility tools
- HTML parsers that don't implement error recovery
- Future browser versions with stricter parsing

## Fix

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