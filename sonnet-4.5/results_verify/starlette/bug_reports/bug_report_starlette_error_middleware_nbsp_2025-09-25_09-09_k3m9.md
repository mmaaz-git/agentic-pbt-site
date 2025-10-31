# Bug Report: starlette.middleware.errors Invalid HTML Entity

**Target**: `starlette.middleware.errors.ServerErrorMiddleware.format_line`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `ServerErrorMiddleware.format_line` method generates invalid HTML by using `&nbsp` instead of `&nbsp;` (missing semicolon) when replacing spaces in error traceback lines.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from starlette.middleware.errors import ServerErrorMiddleware


@given(st.text(min_size=1, max_size=100))
def test_nbsp_has_semicolon(line_text):
    middleware = ServerErrorMiddleware(app=lambda scope, receive, send: None)
    formatted = middleware.format_line(0, line_text, 1, 0)

    if " " in line_text:
        assert "&nbsp;" in formatted, \
            "HTML entity for non-breaking space must end with semicolon"
```

**Failing input**: `"hello world"` (or any string with spaces)

## Reproducing the Bug

```python
from starlette.middleware.errors import ServerErrorMiddleware

middleware = ServerErrorMiddleware(app=lambda scope, receive, send: None)
formatted = middleware.format_line(0, "hello world", 1, 0)

print(formatted)
```

**Output:**
```html
<p class="center-line"><span class="frame-line center-line">
<span class="lineno">1.</span> hello&nbspworld</span></p>
```

Notice `&nbsp` without the semicolon, which is invalid HTML. Valid HTML requires `&nbsp;` with a semicolon.

## Why This Is A Bug

According to HTML specifications, named character references (like non-breaking space) must end with a semicolon. While some browsers may be lenient and render `&nbsp` correctly, it's technically invalid HTML and could cause issues with:
- HTML validators
- Strict HTML parsers
- Screen readers and accessibility tools
- Non-browser HTML consumers

The W3C HTML specification explicitly requires the semicolon for named character references.

## Fix

```diff
diff --git a/starlette/middleware/errors.py b/starlette/middleware/errors.py
index 1234567..abcdefg 100644
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