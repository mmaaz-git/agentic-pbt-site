# Bug Report: starlette.middleware.errors Invalid HTML Entity

**Target**: `starlette.middleware.errors.ServerErrorMiddleware.format_line`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The error middleware's HTML generation uses invalid HTML entities by omitting the required semicolon from `&nbsp`, producing `&nbsp` instead of `&nbsp;`.

## Property-Based Test

```python
from hypothesis import given, settings
import hypothesis.strategies as st
from starlette.middleware.errors import ServerErrorMiddleware


def dummy_app(scope, receive, send):
    pass


@given(
    st.text(min_size=1, max_size=100),
    st.integers(min_value=1, max_value=1000),
    st.integers(min_value=0, max_value=20)
)
@settings(max_examples=300)
def test_format_line_html_entity_correctness(line, frame_lineno, frame_index):
    middleware = ServerErrorMiddleware(dummy_app, debug=True)
    formatted = middleware.format_line(frame_index, line, frame_lineno, frame_index)

    if " " in line and "&nbsp" in formatted:
        assert "&nbsp;" in formatted
```

**Failing input**: `line=' '` (single space)

## Reproducing the Bug

```python
from starlette.middleware.errors import ServerErrorMiddleware


def dummy_app(scope, receive, send):
    pass


middleware = ServerErrorMiddleware(dummy_app, debug=True)
test_line = "    some code with spaces"
formatted = middleware.format_line(5, test_line, 10, 5)

print(formatted)
```

Output:
```html
<p class="center-line"><span class="frame-line center-line">
<span class="lineno">10.</span> &nbsp&nbsp&nbsp&nbspsome&nbspcode&nbspwith&nbspspaces</span></p>
```

Notice `&nbsp` instead of `&nbsp;`.

## Why This Is A Bug

HTML entities must end with a semicolon according to the HTML specification. While most browsers are lenient and will render `&nbsp` correctly, this produces invalid HTML that:
- Fails HTML validators
- May cause issues in strict parsers
- Violates W3C standards

The bug is on line 191 of `errors.py`:
```python
"line": html.escape(line).replace(" ", "&nbsp"),
```

This should be:
```python
"line": html.escape(line).replace(" ", "&nbsp;"),
```

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