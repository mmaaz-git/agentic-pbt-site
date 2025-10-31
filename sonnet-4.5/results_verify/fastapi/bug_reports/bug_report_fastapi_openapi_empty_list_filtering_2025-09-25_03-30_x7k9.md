# Bug Report: fastapi.openapi.utils.get_openapi Empty List Filtering

**Target**: `fastapi.openapi.utils.get_openapi`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `get_openapi` function incorrectly filters out empty list values for `servers` and `tags` parameters, making it impossible to distinguish between "field not provided" and "field explicitly empty".

## Property-Based Test

```python
from hypothesis import given, strategies as st
from fastapi.openapi.utils import get_openapi

@given(
    title=st.text(min_size=1, max_size=100),
    version=st.text(min_size=1, max_size=50),
    servers=st.one_of(st.none(), st.just([]), st.lists(st.fixed_dictionaries({"url": st.just("http://localhost")})))
)
def test_get_openapi_servers_preservation(title, version, servers):
    result = get_openapi(title=title, version=version, servers=servers, routes=[])

    if servers is not None:
        assert "servers" in result
        assert result["servers"] == servers
```

**Failing input**: `servers=[]` or `tags=[]`

## Reproducing the Bug

```python
from fastapi.openapi.utils import get_openapi

result = get_openapi(
    title="Test API",
    version="1.0.0",
    servers=[],
    tags=[],
    routes=[]
)

print(f"Has servers: {'servers' in result}")
print(f"Has tags: {'tags' in result}")
```

Output:
```
Has servers: False
Has tags: False
```

## Why This Is A Bug

According to OpenAPI specification and semantic meaning:
- `servers=None` means "use default servers" (field omitted from output)
- `servers=[]` means "explicitly no servers available" (field present but empty in output)

The current implementation treats empty lists as falsy, making it impossible to distinguish between these two semantically different cases. This is the same bug pattern as the already-reported empty string filtering bug for `description` and `summary` fields.

## Fix

```diff
--- a/fastapi/openapi/utils.py
+++ b/fastapi/openapi/utils.py
@@ -501,7 +501,7 @@ def get_openapi(
         info["license"] = license_info
     output: Dict[str, Any] = {"openapi": openapi_version, "info": info}
-    if servers:
+    if servers is not None:
         output["servers"] = servers
     components: Dict[str, Dict[str, Any]] = {}
     paths: Dict[str, Dict[str, Any]] = {}
@@ -564,7 +564,7 @@ def get_openapi(
     output["paths"] = paths
     if webhook_paths:
         output["webhooks"] = webhook_paths
-    if tags:
+    if tags is not None:
         output["tags"] = tags
     return jsonable_encoder(OpenAPI(**output), by_alias=True, exclude_none=True)
```