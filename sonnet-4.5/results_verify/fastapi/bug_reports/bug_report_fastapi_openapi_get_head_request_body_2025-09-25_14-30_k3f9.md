# Bug Report: FastAPI OpenAPI GET/HEAD Request Body

**Target**: `fastapi.openapi.utils.get_openapi_path` and `fastapi.openapi.constants.METHODS_WITH_BODY`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

FastAPI generates OpenAPI schemas with `requestBody` fields for GET and HEAD requests, violating the OpenAPI 3.x specification which does not support request bodies for these HTTP methods.

## Property-Based Test

```python
from fastapi import FastAPI, Body
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
from hypothesis import given, strategies as st


class Item(BaseModel):
    name: str


@given(method=st.sampled_from(["GET", "HEAD"]))
def test_get_head_should_not_have_request_body_in_openapi(method):
    app = FastAPI()

    if method == "GET":
        @app.get("/test")
        def handler(item: Item = Body(...)):
            return {"item": item}
    else:
        @app.head("/test")
        def handler(item: Item = Body(...)):
            return {"item": item}

    openapi_schema = get_openapi(
        title="Test",
        version="1.0.0",
        routes=app.routes,
    )

    operation = openapi_schema["paths"]["/test"][method.lower()]
    assert "requestBody" not in operation, \
        f"{method} requests should not have requestBody in OpenAPI schema"
```

**Failing input**: Any GET or HEAD route with a Body parameter

## Reproducing the Bug

```python
from fastapi import FastAPI, Body
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel


app = FastAPI()


class Item(BaseModel):
    name: str


@app.get("/test")
def get_with_body(item: Item = Body(...)):
    return {"item": item}


openapi_schema = get_openapi(
    title="Test API",
    version="1.0.0",
    routes=app.routes,
)

print(openapi_schema["paths"]["/test"]["get"])
```

**Output:**
```python
{
    'summary': 'Get With Body',
    'operationId': 'get_with_body_test_get',
    'requestBody': {  # ← This should not be here for GET
        'content': {
            'application/json': {
                'schema': {'$ref': '#/components/schemas/Item'}
            }
        },
        'required': True
    },
    'responses': {...}
}
```

## Why This Is A Bug

According to the OpenAPI 3.x specification:
- GET and HEAD requests do not support `requestBody` fields
- Only POST, PUT, PATCH, DELETE, and OPTIONS methods can have request bodies in OpenAPI schemas

The root cause is in `fastapi/openapi/constants.py`:
```python
METHODS_WITH_BODY = {"GET", "HEAD", "POST", "PUT", "DELETE", "PATCH"}
```

This constant incorrectly includes GET and HEAD, causing `get_openapi_path()` in `fastapi/openapi/utils.py` to generate invalid OpenAPI schemas:
```python
if method in METHODS_WITH_BODY:  # Line 309 in utils.py
    request_body_oai = get_openapi_operation_request_body(...)
    if request_body_oai:
        operation["requestBody"] = request_body_oai
```

## Fix

```diff
--- a/fastapi/openapi/constants.py
+++ b/fastapi/openapi/constants.py
@@ -1,3 +1,3 @@
-METHODS_WITH_BODY = {"GET", "HEAD", "POST", "PUT", "DELETE", "PATCH"}
+METHODS_WITH_BODY = {"POST", "PUT", "DELETE", "PATCH"}
 REF_PREFIX = "#/components/schemas/"
 REF_TEMPLATE = "#/components/schemas/{model}"
```