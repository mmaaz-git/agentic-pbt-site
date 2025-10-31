# Bug Report: fastapi.openapi.utils.get_openapi Empty String Filtering for terms_of_service

**Target**: `fastapi.openapi.utils.get_openapi`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `get_openapi` function incorrectly filters out empty string values for the `terms_of_service` parameter. This is the same bug pattern as the already-reported issue with `description` and `summary` fields.

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
        assert "termsOfService" in result["info"]
        assert result["info"]["termsOfService"] == terms_of_service
```

**Failing input**: `terms_of_service=""`

## Reproducing the Bug

```python
from fastapi.openapi.utils import get_openapi

result = get_openapi(
    title="Test API",
    version="1.0.0",
    terms_of_service="",
    routes=[]
)

print(f"Has termsOfService: {'termsOfService' in result['info']}")
```

Output:
```
Has termsOfService: False
```

## Why This Is A Bug

This is the same issue as the already-reported bug for `description` and `summary` fields. Empty strings are valid OpenAPI values and should be preserved to maintain semantic distinction between:
- `terms_of_service=""` - explicitly empty terms (should be included)
- `terms_of_service=None` - no terms provided (should be omitted)

## Fix

```diff
--- a/fastapi/openapi/utils.py
+++ b/fastapi/openapi/utils.py
@@ -495,7 +495,7 @@ def get_openapi(
         info["summary"] = summary
     if description:
         info["description"] = description
-    if terms_of_service:
+    if terms_of_service is not None:
         info["termsOfService"] = terms_of_service
     if contact:
         info["contact"] = contact
```