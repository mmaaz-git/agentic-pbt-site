# Bug Report: Jinja2Templates.TemplateResponse crashes with context=None

**Target**: `starlette.templating.Jinja2Templates.TemplateResponse`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `TemplateResponse` method crashes with an `AttributeError` when `context=None` is passed, even though the type signature explicitly allows `context: dict[str, Any] | None = None`.

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
```

**Failing input**: Any valid template name (e.g., `"test"`)

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
```

**Expected**: Should create a template response with an empty context dict (since context defaults to None in the signature).

**Actual**: Crashes with `AttributeError: 'NoneType' object has no attribute 'setdefault'` at line 205.

## Why This Is A Bug

The method's type signature (lines 135-157) explicitly allows `context: dict[str, Any] | None = None`, indicating that `None` is a valid value. However, the implementation has multiple issues:

1. **Line 197**: `context = kwargs.get("context", {})` returns `None` when `context=None` is explicitly passed in kwargs (instead of using the default `{}`)

2. **Line 205**: `context.setdefault("request", request)` assumes context is a dict, but crashes when it's `None`

This violates the contract established by the type hints and causes unexpected crashes for valid API usage.

## Fix

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

This fix ensures that when `context=None` is passed, it's converted to an empty dict `{}`, making the behavior consistent with the default parameter behavior and preventing the AttributeError on line 205.