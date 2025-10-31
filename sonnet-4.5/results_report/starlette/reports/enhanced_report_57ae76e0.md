# Bug Report: Jinja2Templates.TemplateResponse crashes when context=None is explicitly passed

**Target**: `starlette.templating.Jinja2Templates.TemplateResponse`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `TemplateResponse` method crashes with an `AttributeError` when `context=None` is explicitly passed, even though the type signature explicitly declares `context: dict[str, Any] | None = None` as valid.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from starlette.templating import Jinja2Templates
from starlette.requests import Request
import tempfile
import os


@given(
    template_name=st.text(
        alphabet=st.characters(min_codepoint=ord('a'), max_codepoint=ord('z')),
        min_size=1,
        max_size=20
    )
)
@settings(max_examples=50)
def test_template_response_none_context_should_not_crash(template_name):
    """
    Property: TemplateResponse should handle context=None gracefully
    since the type signature allows context: dict[str, Any] | None = None
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        template_path = os.path.join(tmpdir, f"{template_name}.html")
        with open(template_path, 'w') as f:
            f.write("<html><body>{{ request.url }}</body></html>")

        templates = Jinja2Templates(directory=tmpdir)

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/test",
            "query_string": b"",
            "headers": [],
            "server": ("localhost", 8000),
        }
        request = Request(scope)

        response = templates.TemplateResponse(
            request=request,
            name=f"{template_name}.html",
            context=None
        )

        assert response is not None
        assert response.status_code == 200


if __name__ == "__main__":
    test_template_response_none_context_should_not_crash()
```

<details>

<summary>
**Failing input**: `template_name='a'` (or any other generated value)
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 49, in <module>
    test_template_response_none_context_should_not_crash()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 9, in test_template_response_none_context_should_not_crash
    template_name=st.text(
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 38, in test_template_response_none_context_should_not_crash
    response = templates.TemplateResponse(
        request=request,
        name=f"{template_name}.html",
        context=None
    )
  File "/home/npc/miniconda/lib/python3.13/site-packages/starlette/templating.py", line 197, in TemplateResponse
    request = kwargs.get("request", context.get("request"))
                                    ^^^^^^^^^^^
AttributeError: 'NoneType' object has no attribute 'get'
Falsifying example: test_template_response_none_context_should_not_crash(
    template_name='a',  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from starlette.templating import Jinja2Templates
from starlette.requests import Request
import tempfile
import os

with tempfile.TemporaryDirectory() as tmpdir:
    with open(os.path.join(tmpdir, "test.html"), 'w') as f:
        f.write("<html><body>{{ request.url }}</body></html>")

    templates = Jinja2Templates(directory=tmpdir)

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "query_string": b"",
        "headers": [],
        "server": ("localhost", 8000),
    }
    request = Request(scope)

    response = templates.TemplateResponse(
        request=request,
        name="test.html",
        context=None
    )

    print("Response created successfully")
    print(f"Status code: {response.status_code}")
```

<details>

<summary>
AttributeError: 'NoneType' object has no attribute 'get' at line 198
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/37/repo.py", line 22, in <module>
    response = templates.TemplateResponse(
        request=request,
        name="test.html",
        context=None
    )
  File "/home/npc/miniconda/lib/python3.13/site-packages/starlette/templating.py", line 197, in TemplateResponse
    request = kwargs.get("request", context.get("request"))
                                    ^^^^^^^^^^^
AttributeError: 'NoneType' object has no attribute 'get'
```
</details>

## Why This Is A Bug

This violates the explicit API contract established by the type hints. The method has two overloaded signatures (lines 135-144 and 147-157 in starlette/templating.py), both declaring `context: dict[str, Any] | None = None`. This type annotation explicitly indicates that `None` is an acceptable value for the context parameter.

The bug occurs because of incorrect handling of the None case in the implementation. When processing kwargs-only arguments (line 197), the code uses `context = kwargs.get("context", {})`. However, when `context=None` is explicitly passed in kwargs, the key "context" exists with value None, so `dict.get()` returns None instead of the default `{}`. This leads to an immediate crash on line 198 where `context.get("request")` is called on None.

The error is actually earlier than originally reported in the initial analysis. It occurs at line 198 (`request = kwargs.get("request", context.get("request"))`) rather than line 205 (`context.setdefault("request", request)`). The crash happens because `context` is None and the code attempts to call `.get()` on it.

Users reasonably expect that passing `context=None` explicitly should behave the same as omitting the parameter entirely (both using an empty dict as context). This expectation is reinforced by the type signature showing None as both an acceptable type and the default value.

## Relevant Context

The Starlette documentation at https://www.starlette.io/templates/ states that "the incoming `request` instance must be included as part of the template context" but doesn't explicitly address None handling. The documentation examples show either passing a dict with the request or omitting the context parameter entirely.

The implementation has different code paths:
- Lines 159-186: Handles positional arguments (both old and new style)
- Lines 187-204: Handles kwargs-only arguments (where the bug occurs)

The bug only affects the kwargs-only code path when `context=None` is explicitly passed. The positional arguments path (line 182) correctly uses `kwargs.get("context", {})` in a context where it works properly.

Type checkers like mypy would validate code passing `context=None` as correct based on the type hints, but the runtime behavior contradicts this validation.

## Proposed Fix

```diff
--- a/starlette/templating.py
+++ b/starlette/templating.py
@@ -194,7 +194,7 @@ class Jinja2Templates:
                 if "request" not in kwargs.get("context", {}):
                     raise ValueError('context must include a "request" key')

-            context = kwargs.get("context", {})
+            context = kwargs.get("context") or {}
             request = kwargs.get("request", context.get("request"))
             name = cast(str, kwargs["name"])
             status_code = kwargs.get("status_code", 200)
```