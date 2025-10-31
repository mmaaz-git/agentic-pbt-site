# Bug Report: starlette.middleware.errors Missing Semicolon in HTML Entity

**Target**: `starlette.middleware.errors.ServerErrorMiddleware.format_line`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `ServerErrorMiddleware.format_line` method generates invalid HTML by using `&nbsp` instead of `&nbsp;` (missing semicolon) when replacing spaces in error traceback lines, violating W3C HTML5 specifications.

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

<details>

<summary>
**Failing input**: `' '`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/26/hypo.py", line 16, in <module>
    test_nbsp_has_semicolon()
    ~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/26/hypo.py", line 6, in test_nbsp_has_semicolon
    def test_nbsp_has_semicolon(line_text):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/26/hypo.py", line 11, in test_nbsp_has_semicolon
    assert "&nbsp;" in formatted, \
           ^^^^^^^^^^^^^^^^^^^^^
AssertionError: HTML entity for non-breaking space must end with semicolon
Falsifying example: test_nbsp_has_semicolon(
    line_text=' ',
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/26/hypo.py:11
```
</details>

## Reproducing the Bug

```python
from starlette.middleware.errors import ServerErrorMiddleware

# Create an instance of ServerErrorMiddleware
middleware = ServerErrorMiddleware(app=lambda scope, receive, send: None)

# Test the format_line method with a string containing spaces
test_line = "hello world"
formatted = middleware.format_line(0, test_line, 1, 0)

print("Input line:", repr(test_line))
print("\nFormatted output:")
print(formatted)
print("\nSearching for '&nbsp' (without semicolon):", "&nbsp" in formatted and "&nbsp;" not in formatted)
print("Searching for '&nbsp;' (with semicolon):", "&nbsp;" in formatted)
```

<details>

<summary>
Invalid HTML entity generated - missing semicolon in &nbsp
</summary>
```
Input line: 'hello world'

Formatted output:

<p class="center-line"><span class="frame-line center-line">
<span class="lineno">1.</span> hello&nbspworld</span></p>


Searching for '&nbsp' (without semicolon): True
Searching for '&nbsp;' (with semicolon): False
```
</details>

## Why This Is A Bug

This violates the W3C HTML5 specification which explicitly requires all named character references to end with a semicolon. According to both W3C and WHATWG standards, character references without trailing semicolons are "officially forbidden for HTML authors to use."

While many browsers are lenient and will still render `&nbsp` correctly as a backward compatibility feature, this produces technically invalid HTML that:
- Fails HTML validation (validators flag this as an error: "Character reference was not terminated by a semicolon")
- May cause issues with strict HTML parsers
- Could break accessibility tools or screen readers
- May fail in non-browser HTML consumers
- Contradicts the function's clear intent to produce valid HTML (evidenced by its use of `html.escape()`)

The bug occurs in debug mode error pages when displaying Python traceback with indented code, where spaces need to be preserved using non-breaking spaces.

## Relevant Context

The bug is located in `/home/npc/pbt/agentic-pbt/envs/starlette_env/lib/python3.13/site-packages/starlette/middleware/errors.py` at line 191. The `format_line` method is responsible for formatting individual lines of Python code in error tracebacks displayed in debug mode.

The method already demonstrates awareness of HTML encoding requirements by using `html.escape()` to handle special characters like `<` and `>`, making the missing semicolon appear to be an oversight rather than intentional.

W3C HTML validator documentation: https://validator.w3.org/
HTML5 specification on character references: https://html.spec.whatwg.org/multipage/syntax.html#character-references

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