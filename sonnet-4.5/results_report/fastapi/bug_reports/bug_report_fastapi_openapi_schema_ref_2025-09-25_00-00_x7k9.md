# Bug Report: fastapi.openapi.models.Schema ref Field Attribute Access

**Target**: `fastapi.openapi.models.Schema`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `ref` field in the `Schema` model cannot be accessed after setting it using the field name in the constructor. When creating `Schema(ref='value')`, the value is stored internally (visible in `model_dump()`) but accessing `schema.ref` returns `None` instead of the stored value.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from fastapi.openapi.models import Schema

@given(st.text(min_size=1))
@settings(max_examples=500)
def test_schema_ref_field_roundtrip(ref_value):
    schema = Schema(ref=ref_value)
    assert schema.ref == ref_value
```

**Failing input**: `ref_value='0'` (or any string value)

## Reproducing the Bug

```python
from fastapi.openapi.models import Schema

schema = Schema(ref='#/components/schemas/MyModel')

print(f"schema.ref = {repr(schema.ref)}")

print(f"model_dump() = {schema.model_dump(exclude_none=True)}")

assert schema.ref == '#/components/schemas/MyModel'
```

**Output:**
```
schema.ref = None
model_dump() = {'ref': '#/components/schemas/MyModel'}
AssertionError: None != '#/components/schemas/MyModel'
```

## Why This Is A Bug

The `ref` field is defined with an alias `$ref` in the Schema model:

```python
ref: Optional[str] = Field(default=None, alias="$ref")
```

When users create a Schema using the Python field name (`ref='value'`), they expect to be able to read it back using `schema.ref`. However, the value returns `None` even though:
1. The value is stored (visible in `model_dump()`)
2. The value can be accessed if created using the alias: `Schema(**{'$ref': 'value'})`

This inconsistency violates the principle of least surprise and will cause bugs in any code that creates Schema objects using the field name and then tries to read the value back.

## Fix

The `BaseModelWithConfig` class (which `Schema` inherits from) needs to add `populate_by_name=True` to its configuration to allow using both the field name and the alias:

```diff
--- a/fastapi/openapi/models.py
+++ b/fastapi/openapi/models.py
@@ -58,10 +58,14 @@ class BaseModelWithConfig(BaseModel):
     if PYDANTIC_V2:
-        model_config = {"extra": "allow"}
+        from pydantic import ConfigDict
+        model_config = ConfigDict(extra="allow", populate_by_name=True)

     else:

         class Config:
             extra = "allow"
+            allow_population_by_field_name = True
```

This allows Pydantic models with aliased fields to accept values using either the field name or the alias, and properly store them for attribute access.