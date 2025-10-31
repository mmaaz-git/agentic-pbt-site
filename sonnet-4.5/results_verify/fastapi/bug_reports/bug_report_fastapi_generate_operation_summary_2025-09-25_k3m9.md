# Bug Report: FastAPI generate_operation_summary Empty String Handling

**Target**: `fastapi.openapi.utils.generate_operation_summary`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When a route explicitly sets `summary=""` (empty string), `generate_operation_summary()` incorrectly treats it as falsy and falls back to auto-generating a summary from the route name, instead of respecting the explicit empty string.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from fastapi.openapi.utils import generate_operation_summary
from unittest.mock import Mock

@given(st.text())
def test_generate_operation_summary_uses_provided_summary(summary):
    route = Mock()
    route.summary = summary
    route.name = "some_name"
    result = generate_operation_summary(route=route, method="get")
    assert result == summary
```

**Failing input**: `summary=''`

## Reproducing the Bug

```python
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

app = FastAPI()

@app.get("/items", summary="")
def read_items():
    return {"items": []}

openapi_schema = get_openapi(
    title="Test API",
    version="1.0.0",
    routes=app.routes,
)

print(openapi_schema['paths']['/items']['get']['summary'])
```

Expected: `''`
Actual: `'Read Items'`

## Why This Is A Bug

The function uses a truthy check (`if route.summary:`) instead of an explicit None check (`if route.summary is not None:`). This means:

1. Empty strings are valid strings and should be respected when explicitly set
2. The type annotation `Optional[str]` allows empty strings
3. A user who explicitly sets `summary=""` expects that value to be used, not ignored
4. The function should only fall back to auto-generation when summary is `None`

## Fix

```diff
--- a/fastapi/openapi/utils.py
+++ b/fastapi/openapi/utils.py
@@ -221,6 +221,6 @@ def generate_operation_id(


 def generate_operation_summary(*, route: routing.APIRoute, method: str) -> str:
-    if route.summary:
+    if route.summary is not None:
         return route.summary
     return route.name.replace("_", " ").title()
```