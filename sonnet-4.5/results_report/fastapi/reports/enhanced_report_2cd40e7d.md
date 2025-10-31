# Bug Report: fastapi.openapi.utils.get_openapi Empty List Filtering

**Target**: `fastapi.openapi.utils.get_openapi`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `get_openapi` function incorrectly filters out empty list values for `servers` and `tags` parameters, making it impossible to distinguish between "field not provided" (None) and "field explicitly empty" ([]).

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
        assert "servers" in result, f"servers field should be present when servers={servers}"
        assert result["servers"] == servers, f"servers value should be {servers} but got {result.get('servers')}"


@given(
    title=st.text(min_size=1, max_size=100),
    version=st.text(min_size=1, max_size=50),
    tags=st.one_of(st.none(), st.just([]), st.lists(st.fixed_dictionaries({"name": st.text(min_size=1, max_size=50)})))
)
def test_get_openapi_tags_preservation(title, version, tags):
    result = get_openapi(title=title, version=version, tags=tags, routes=[])

    if tags is not None:
        assert "tags" in result, f"tags field should be present when tags={tags}"
        assert result["tags"] == tags, f"tags value should be {tags} but got {result.get('tags')}"


if __name__ == "__main__":
    print("Testing servers preservation...")
    test_get_openapi_servers_preservation()
    print()

    print("Testing tags preservation...")
    test_get_openapi_tags_preservation()
    print()

    print("All tests passed if no output above!")
```

<details>

<summary>
**Failing input**: `servers=[]` and `tags=[]`
</summary>
```
Testing servers preservation...
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 32, in <module>
    test_get_openapi_servers_preservation()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 5, in test_get_openapi_servers_preservation
    title=st.text(min_size=1, max_size=100),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 13, in test_get_openapi_servers_preservation
    assert "servers" in result, f"servers field should be present when servers={servers}"
           ^^^^^^^^^^^^^^^^^^^
AssertionError: servers field should be present when servers=[]
Falsifying example: test_get_openapi_servers_preservation(
    title='0',  # or any other generated value
    version='0',  # or any other generated value
    servers=[],
)
```
</details>

## Reproducing the Bug

```python
from fastapi.openapi.utils import get_openapi

# Test case 1: Empty lists for servers and tags
result = get_openapi(
    title="Test API",
    version="1.0.0",
    servers=[],
    tags=[],
    routes=[]
)

print("Test with empty lists:")
print(f"Has servers: {'servers' in result}")
print(f"Has tags: {'tags' in result}")
print()

# Test case 2: None values for servers and tags
result_none = get_openapi(
    title="Test API",
    version="1.0.0",
    servers=None,
    tags=None,
    routes=[]
)

print("Test with None values:")
print(f"Has servers: {'servers' in result_none}")
print(f"Has tags: {'tags' in result_none}")
print()

# Test case 3: Non-empty lists
result_nonempty = get_openapi(
    title="Test API",
    version="1.0.0",
    servers=[{"url": "http://localhost:8000"}],
    tags=[{"name": "test", "description": "Test tag"}],
    routes=[]
)

print("Test with non-empty lists:")
print(f"Has servers: {'servers' in result_nonempty}")
print(f"Servers value: {result_nonempty.get('servers', 'NOT PRESENT')}")
print(f"Has tags: {'tags' in result_nonempty}")
print(f"Tags value: {result_nonempty.get('tags', 'NOT PRESENT')}")
```

<details>

<summary>
Empty lists are incorrectly filtered out, making them indistinguishable from None values
</summary>
```
Test with empty lists:
Has servers: False
Has tags: False

Test with None values:
Has servers: False
Has tags: False

Test with non-empty lists:
Has servers: True
Servers value: [{'url': 'http://localhost:8000'}]
Has tags: True
Tags value: [{'name': 'test', 'description': 'Test tag'}]
```
</details>

## Why This Is A Bug

According to the OpenAPI specification 3.1.0, the `servers` and `tags` fields have specific semantic meanings:

1. **servers field**: When omitted (None), the default server URL is implied to be `/`. When explicitly set to an empty array `[]`, it indicates that no servers are available for this API - this is semantically different from using the default.

2. **tags field**: When omitted (None), no grouping is applied. When explicitly set to an empty array `[]`, it indicates that tag grouping has been considered but no tags are defined - useful for programmatically generated specs.

The current implementation uses Python's truthiness check (`if servers:` on line 504 and `if tags:` on line 566) which treats empty lists as falsy, making it impossible to distinguish between:
- `servers=None` → "use default server configuration" (field omitted)
- `servers=[]` → "explicitly no servers available" (field present but empty)

This violates the principle of explicit configuration and prevents users from accurately representing their API's intended behavior in the OpenAPI specification.

## Relevant Context

This bug follows the same pattern as other empty value filtering issues in FastAPI's OpenAPI generation. The OpenAPI specification (https://spec.openapis.org/oas/v3.1.0) explicitly allows empty arrays for both `servers` and `tags` fields, and these have different meanings than omitting the fields entirely.

The relevant code is in `/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages/fastapi/openapi/utils.py`:
- Line 504-505: `if servers: output["servers"] = servers`
- Line 566-567: `if tags: output["tags"] = tags`

## Proposed Fix

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
@@ -563,7 +563,7 @@ def get_openapi(
     output["paths"] = paths
     if webhook_paths:
         output["webhooks"] = webhook_paths
-    if tags:
+    if tags is not None:
         output["tags"] = tags
     return jsonable_encoder(OpenAPI(**output), by_alias=True, exclude_none=True)
```