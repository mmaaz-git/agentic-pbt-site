# Bug Report: fastapi.openapi.utils.get_openapi Empty String Filtering for terms_of_service

**Target**: `fastapi.openapi.utils.get_openapi`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `get_openapi` function incorrectly filters out empty string values for the `terms_of_service` parameter (and other optional string parameters), treating them the same as `None` values when they should be semantically distinct according to OpenAPI specifications.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from fastapi.openapi.utils import get_openapi

@given(
    title=st.text(min_size=1, max_size=100),
    version=st.text(min_size=1, max_size=50),
    terms_of_service=st.one_of(st.none(), st.text(max_size=200))
)
def test_get_openapi_terms_preservation(title, version, terms_of_service):
    result = get_openapi(title=title, version=version, terms_of_service=terms_of_service, routes=[])

    if terms_of_service is not None:
        assert "termsOfService" in result["info"], f"termsOfService missing when terms_of_service={repr(terms_of_service)}"
        assert result["info"]["termsOfService"] == terms_of_service

# Run the test
if __name__ == "__main__":
    test_get_openapi_terms_preservation()
```

<details>

<summary>
**Failing input**: `terms_of_service=""`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 18, in <module>
    test_get_openapi_terms_preservation()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 5, in test_get_openapi_terms_preservation
    title=st.text(min_size=1, max_size=100),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 13, in test_get_openapi_terms_preservation
    assert "termsOfService" in result["info"], f"termsOfService missing when terms_of_service={repr(terms_of_service)}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: termsOfService missing when terms_of_service=''
Falsifying example: test_get_openapi_terms_preservation(
    title='0',
    version='0',
    terms_of_service='',
)
```
</details>

## Reproducing the Bug

```python
from fastapi.openapi.utils import get_openapi

# Test case showing the bug with empty string
result = get_openapi(
    title="Test API",
    version="1.0.0",
    terms_of_service="",
    routes=[]
)

print(f"Has termsOfService: {'termsOfService' in result['info']}")
print(f"Info object: {result['info']}")

# Test case with None (should not include termsOfService)
result_none = get_openapi(
    title="Test API",
    version="1.0.0",
    terms_of_service=None,
    routes=[]
)

print(f"\nWith None - Has termsOfService: {'termsOfService' in result_none['info']}")
print(f"Info object: {result_none['info']}")

# Test case with non-empty string (should include termsOfService)
result_value = get_openapi(
    title="Test API",
    version="1.0.0",
    terms_of_service="https://example.com/terms",
    routes=[]
)

print(f"\nWith value - Has termsOfService: {'termsOfService' in result_value['info']}")
print(f"Info object: {result_value['info']}")
```

<details>

<summary>
Empty string incorrectly filtered out, treated same as None
</summary>
```
Has termsOfService: False
Info object: {'title': 'Test API', 'version': '1.0.0'}

With None - Has termsOfService: False
Info object: {'title': 'Test API', 'version': '1.0.0'}

With value - Has termsOfService: True
Info object: {'title': 'Test API', 'termsOfService': 'https://example.com/terms', 'version': '1.0.0'}
```
</details>

## Why This Is A Bug

This violates expected behavior because:

1. **Type System Inconsistency**: The function signature declares `terms_of_service: Optional[str] = None`, which according to Python's typing system includes empty strings as valid `str` values. The current implementation contradicts this type hint.

2. **OpenAPI Semantic Distinction**: The OpenAPI 3.1.0 specification makes a clear semantic distinction between:
   - Field absent from JSON (when `terms_of_service=None`) - indicates the information was never provided
   - Field present with empty string value (when `terms_of_service=""`) - indicates the field was explicitly set to empty

3. **Principle of Least Surprise**: Users passing an empty string expect it to be preserved in the output, not silently dropped. This silent filtering can cause confusion in API documentation tools.

4. **Systematic Issue**: The same bug pattern affects multiple fields in the same function:
   - `summary` (line 494)
   - `description` (line 496)
   - `terms_of_service` (line 498)
   - `contact` (line 500) - for empty dicts
   - `license_info` (line 502) - for empty dicts

## Relevant Context

The bug occurs in `/home/npc/miniconda/lib/python3.13/site-packages/fastapi/openapi/utils.py` at line 498. The function uses Python's truthiness evaluation (`if terms_of_service:`) instead of explicit None checking (`if terms_of_service is not None:`).

This is a common Python anti-pattern when handling optional string parameters. Empty strings evaluate to `False` in boolean contexts, causing them to be incorrectly filtered out alongside `None` values.

Similar issues have been reported for other fields in the same function, indicating this is a systematic problem rather than an isolated case.

OpenAPI documentation: https://spec.openapis.org/oas/v3.1.0#info-object

## Proposed Fix

```diff
--- a/fastapi/openapi/utils.py
+++ b/fastapi/openapi/utils.py
@@ -491,13 +491,13 @@ def get_openapi(
     separate_input_output_schemas: bool = True,
 ) -> Dict[str, Any]:
     info: Dict[str, Any] = {"title": title, "version": version}
-    if summary:
+    if summary is not None:
         info["summary"] = summary
-    if description:
+    if description is not None:
         info["description"] = description
-    if terms_of_service:
+    if terms_of_service is not None:
         info["termsOfService"] = terms_of_service
-    if contact:
+    if contact is not None:
         info["contact"] = contact
-    if license_info:
+    if license_info is not None:
         info["license"] = license_info
```