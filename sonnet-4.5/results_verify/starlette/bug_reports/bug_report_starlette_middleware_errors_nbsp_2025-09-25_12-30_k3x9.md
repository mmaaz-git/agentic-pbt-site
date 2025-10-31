# Bug Report: starlette.middleware.errors Malformed HTML Entity

**Target**: `starlette.middleware.errors.ServerErrorMiddleware.format_line`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `ServerErrorMiddleware.format_line` method generates malformed HTML by using `&nbsp` without the trailing semicolon, which should be `&nbsp;` according to HTML standards.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/starlette_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from starlette.middleware.errors import ServerErrorMiddleware


def dummy_app(scope, receive, send):
    pass


@given(st.text(min_size=1, max_size=100))
def test_format_line_nbsp_entity(line_text):
    middleware = ServerErrorMiddleware(dummy_app, debug=True)

    formatted = middleware.format_line(
        index=0,
        line=line_text,
        frame_lineno=10,
        frame_index=0
    )

    if " " in line_text:
        if "&nbsp" in formatted and "&nbsp;" not in formatted:
            raise AssertionError("HTML entity &nbsp is missing trailing semicolon")
```

**Failing input**: `line_text=' '` (single space)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/starlette_env/lib/python3.13/site-packages')

from starlette.middleware.errors import ServerErrorMiddleware


def dummy_app(scope, receive, send):
    pass


middleware = ServerErrorMiddleware(dummy_app, debug=True)
result = middleware.format_line(index=0, line="hello world", frame_lineno=10, frame_index=0)

print(result)
```

Output contains `&nbsp` instead of `&nbsp;`:
```html
<p class="center-line"><span class="frame-line center-line">
<span class="lineno">10.</span> hello&nbspworld</span></p>
```

## Why This Is A Bug

According to HTML specifications, named character references should end with a semicolon. While some browsers may be lenient and parse `&nbsp` correctly, it is technically malformed HTML and can lead to:
- HTML validation errors
- Potential parsing ambiguities in strict parsers
- Inconsistent behavior across different HTML parsers

The HTML5 specification explicitly requires the semicolon for named character references.

## Fix

```diff
--- a/starlette/middleware/errors.py
+++ b/starlette/middleware/errors.py
@@ -188,7 +188,7 @@ class ServerErrorMiddleware:
     def format_line(self, index: int, line: str, frame_lineno: int, frame_index: int) -> str:
         values = {
-            "line": html.escape(line).replace(" ", "&nbsp"),
+            "line": html.escape(line).replace(" ", "&nbsp;"),
             "lineno": (frame_lineno - frame_index) + index,
         }
```