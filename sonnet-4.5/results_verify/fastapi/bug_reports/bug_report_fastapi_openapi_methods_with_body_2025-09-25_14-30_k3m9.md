# Bug Report: fastapi.openapi GET and HEAD Request Bodies

**Target**: `fastapi.openapi.constants.METHODS_WITH_BODY`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The constant `METHODS_WITH_BODY` incorrectly includes HTTP methods GET and HEAD, allowing request bodies to be documented in the OpenAPI schema for these methods. This violates HTTP best practices as defined in RFC 7231 and goes against OpenAPI 3.1.0 recommendations.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from fastapi import FastAPI, Body
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel


class Item(BaseModel):
    name: str


@given(st.sampled_from(["GET", "HEAD"]))
def test_http_methods_should_not_support_request_bodies(method):
    app = FastAPI()

    if method == "GET":
        @app.get("/items")
        def endpoint(item: Item = Body(...)):
            return item
    else:
        @app.head("/items")
        def endpoint(item: Item = Body(...)):
            return None

    openapi_schema = get_openapi(
        title="Test",
        version="1.0.0",
        routes=app.routes,
    )

    operation = openapi_schema["paths"]["/items"][method.lower()]
    assert "requestBody" not in operation, \
        f"{method} requests should not have requestBody in OpenAPI schema"
```

**Failing input**: Any GET or HEAD request with a Body parameter fails the assertion.

## Reproducing the Bug

```python
from fastapi import FastAPI, Body
from fastapi.openapi.utils import get_openapi
from fastapi.openapi.constants import METHODS_WITH_BODY
from pydantic import BaseModel
import json


class Item(BaseModel):
    name: str


print("METHODS_WITH_BODY =", METHODS_WITH_BODY)

app = FastAPI()

@app.get("/items")
def get_items(item: Item = Body(...)):
    return {"item": item}

openapi_schema = get_openapi(
    title="Test API",
    version="1.0.0",
    routes=app.routes,
)

get_operation = openapi_schema["paths"]["/items"]["get"]
print("\nGET operation has requestBody:")
print(json.dumps(get_operation.get("requestBody"), indent=2))
```

**Output:**
```
METHODS_WITH_BODY = {'DELETE', 'PUT', 'POST', 'GET', 'HEAD', 'PATCH'}

GET operation has requestBody:
{
  "content": {
    "application/json": {
      "schema": {
        "$ref": "#/components/schemas/Item"
      }
    }
  },
  "required": true
}
```

## Why This Is A Bug

1. **HTTP RFC 7231 Violation**: According to RFC 7231 Section 4.3.1: "A payload within a GET request message has no defined semantics." Similarly, HEAD requests should not have payloads.

2. **OpenAPI 3.1.0 Recommendations**: The OpenAPI specification states that for GET and HEAD, "requestBody is permitted but does not have well-defined semantics and SHOULD be avoided if possible."

3. **Misleading Constant Name**: The constant `METHODS_WITH_BODY` implies these methods should have request bodies, when in fact:
   - **Well-defined semantics**: POST, PUT, PATCH
   - **No/vague semantics**: GET, HEAD, DELETE

4. **Violates HTTP Best Practices**: Most HTTP clients and proxies do not support or properly handle request bodies in GET/HEAD requests. Including them in the OpenAPI schema suggests they are supported, which can lead to:
   - Client implementation errors
   - Proxy/cache issues
   - Interoperability problems

5. **Contract Violation**: The constant name and its usage create a false expectation that GET and HEAD are appropriate methods for operations with request bodies.

## Fix

```diff
--- a/fastapi/openapi/constants.py
+++ b/fastapi/openapi/constants.py
@@ -1,4 +1,6 @@
-METHODS_WITH_BODY = {"GET", "HEAD", "POST", "PUT", "DELETE", "PATCH"}
+# HTTP methods that have well-defined semantics for request bodies according to RFC 7231
+# GET, HEAD, and DELETE are excluded as they have no/vague request body semantics
+METHODS_WITH_BODY = {"POST", "PUT", "PATCH"}
 REF_PREFIX = "#/components/schemas/"
 REF_TEMPLATE = "#/components/schemas/{model}"
```

**Note**: DELETE is also excluded in this fix because RFC 7231 does not define semantics for DELETE request bodies, though some APIs use them. If DELETE support is desired, it should be documented and considered carefully.