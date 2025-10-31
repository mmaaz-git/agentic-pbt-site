# Bug Report: fastapi.openapi ServerVariable Default Validation

**Target**: `fastapi.openapi.models.ServerVariable`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `ServerVariable` model does not validate that the `default` value is one of the `enum` values when an `enum` is provided, violating the OpenAPI 3.1.0 specification requirement.

## Property-Based Test

```python
from fastapi.openapi.models import ServerVariable
from hypothesis import given, strategies as st, settings


@given(
    st.lists(st.text(min_size=1, max_size=20), min_size=2, max_size=10),
    st.text(min_size=1, max_size=20)
)
@settings(max_examples=200)
def test_server_variable_default_not_validated(enum_values, default):
    enum_values = list(set(enum_values))
    if len(enum_values) < 2:
        return

    if default in enum_values:
        return

    sv = ServerVariable(
        enum=enum_values,
        default=default
    )

    assert sv.default == default
    assert sv.default not in sv.enum
```

**Failing input**: `enum=["production", "staging"], default="invalid"`

## Reproducing the Bug

```python
from fastapi.openapi.models import ServerVariable

sv = ServerVariable(
    enum=["production", "staging", "development"],
    default="invalid_environment"
)

print(f"enum: {sv.enum}")
print(f"default: {sv.default}")
print(f"Is default in enum? {sv.default in sv.enum}")
```

Output:
```
enum: ['production', 'staging', 'development']
default: invalid_environment
Is default in enum? False
```

This should raise a `ValidationError` but doesn't.

## Why This Is A Bug

According to the [OpenAPI 3.1.0 Specification](https://spec.openapis.org/oas/v3.1.0#server-variable-object), for the `default` field in a Server Variable Object:

> "If the `enum` is defined, the value _MUST_ exist in the enum's values."

The current implementation allows creating a `ServerVariable` with a `default` value that is not in the `enum`, which violates the specification and could lead to invalid OpenAPI documents being generated.

## Fix

Add a Pydantic validator to the `ServerVariable` model to enforce this constraint:

```diff
--- a/fastapi/openapi/models.py
+++ b/fastapi/openapi/models.py
@@ -1,6 +1,7 @@
 from enum import Enum
 from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Type, Union

+from pydantic import field_validator
 from fastapi._compat import (
     PYDANTIC_V2,
     CoreSchema,
@@ -89,6 +90,14 @@ class Info(BaseModelWithConfig):

 class ServerVariable(BaseModelWithConfig):
     enum: Annotated[Optional[List[str]], Field(min_length=1)] = None
     default: str
     description: Optional[str] = None
+
+    @field_validator("default")
+    @classmethod
+    def validate_default_in_enum(cls, v: str, info) -> str:
+        enum = info.data.get("enum")
+        if enum is not None and v not in enum:
+            raise ValueError(f"default value '{v}' must be one of enum values: {enum}")
+        return v
```