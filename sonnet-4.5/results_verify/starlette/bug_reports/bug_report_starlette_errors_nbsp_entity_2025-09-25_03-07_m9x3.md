# Bug Report: starlette.middleware.errors Incomplete HTML Entity

**Target**: `starlette.middleware.errors.ServerErrorMiddleware.format_line`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `ServerErrorMiddleware.format_line` method uses incomplete HTML entity `&nbsp` (without semicolon) instead of the proper `&nbsp;` when replacing spaces in error traceback output.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from starlette.middleware.errors import ServerErrorMiddleware


@given(line=st.text(min_size=1, max_size=100).filter(lambda x: " " in x))
@settings(max_examples=100)
def test_format_line_proper_html_entities(line):
    middleware = ServerErrorMiddleware(app=lambda: None)

    formatted = middleware.format_line(
        index=0,
        line=line,
        frame_lineno=10,
        frame_index=0
    )

    assert "&nbsp;" in formatted or "&nbsp" not in formatted, \
        "HTML entity &nbsp should end with semicolon"
```

**Failing input**: Any line containing spaces, e.g., `"hello world"`

## Reproducing the Bug

```python
from starlette.middleware.errors import ServerErrorMiddleware

middleware = ServerErrorMiddleware(app=lambda: None)
line = "hello world"
formatted = middleware.format_line(
    index=0,
    line=line,
    frame_lineno=10,
    frame_index=0
)
print(formatted)
```

**Output**: Contains `hello&nbspworld` (incomplete entity)

**Expected**: Should contain `hello&nbsp;world` (with semicolon)

## Why This Is A Bug

HTML entities should end with a semicolon according to the HTML specification. While most browsers are lenient and render `&nbsp` correctly, it is not valid HTML. This can cause issues with:
- HTML validators reporting errors
- Strict HTML parsers
- Automated HTML processing tools
- Screen readers and accessibility tools

The code violates HTML standards and could cause issues in certain contexts.

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