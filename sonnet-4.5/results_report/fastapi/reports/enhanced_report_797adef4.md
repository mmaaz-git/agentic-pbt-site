# Bug Report: fastapi.openapi.models.Schema ref Field Inaccessible After Construction

**Target**: `fastapi.openapi.models.Schema`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `ref` field in FastAPI's `Schema` model cannot be accessed after setting it using the field name in the constructor. When creating `Schema(ref='value')`, the value is stored as an extra field but `schema.ref` returns `None` instead of the stored value.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from fastapi.openapi.models import Schema

@given(st.text(min_size=1))
@settings(max_examples=500)
def test_schema_ref_field_roundtrip(ref_value):
    schema = Schema(ref=ref_value)
    assert schema.ref == ref_value

if __name__ == "__main__":
    test_schema_ref_field_roundtrip()
```

<details>

<summary>
**Failing input**: `ref_value='0'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/11/hypo.py", line 11, in <module>
    test_schema_ref_field_roundtrip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/11/hypo.py", line 5, in test_schema_ref_field_roundtrip
    @settings(max_examples=500)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/11/hypo.py", line 8, in test_schema_ref_field_roundtrip
    assert schema.ref == ref_value
           ^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_schema_ref_field_roundtrip(
    ref_value='0',  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from fastapi.openapi.models import Schema

# Create Schema using field name
schema = Schema(ref='#/components/schemas/MyModel')

# Try to access ref field
print(f"schema.ref = {repr(schema.ref)}")

# Show that the value is stored internally
print(f"model_dump() = {schema.model_dump(exclude_none=True)}")

# This assertion will fail
assert schema.ref == '#/components/schemas/MyModel', f"Expected '#/components/schemas/MyModel' but got {repr(schema.ref)}"
```

<details>

<summary>
AssertionError: Expected '#/components/schemas/MyModel' but got None
</summary>
```
schema.ref = None
model_dump() = {'ref': '#/components/schemas/MyModel'}
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/11/repo.py", line 13, in <module>
    assert schema.ref == '#/components/schemas/MyModel', f"Expected '#/components/schemas/MyModel' but got {repr(schema.ref)}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Expected '#/components/schemas/MyModel' but got None
```
</details>

## Why This Is A Bug

The `ref` field in the Schema model is defined with an alias `$ref` to match the OpenAPI/JSON Schema specification:

```python
ref: Optional[str] = Field(default=None, alias="$ref")
```

In Pydantic V2, when a field has an alias and `populate_by_name` is not set to `True`, the field can only be populated using the alias during validation. However, FastAPI's `BaseModelWithConfig` class has `extra="allow"` configured, which causes an unexpected behavior:

1. When users create `Schema(ref='value')`, Pydantic doesn't recognize `ref` as the aliased field
2. Instead, it stores `ref` as an extra field in `model_extra`
3. This makes `schema.ref` return `None` (the default value) rather than the provided value
4. The value is visible in `model_dump()` but inaccessible via attribute access

This violates the principle of least surprise - if a parameter is accepted in the constructor without error, users expect to be able to access it via the corresponding attribute. The current behavior will cause silent failures in production code where developers create Schema objects and then try to read the ref field.

## Relevant Context

- FastAPI uses Pydantic V2 in the tested environment (version 2.10.3)
- The `BaseModelWithConfig` class at line 58-60 of `/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages/fastapi/openapi/models.py` only sets `extra="allow"`
- The same issue affects other aliased fields in the Schema model: `schema_`, `vocabulary`, `id`, `anchor`, `dynamicAnchor`, `dynamicRef`, `defs`, `comment`, `not_`, `if_`, `else_`
- The PathItem class (line 316) also has a `ref` field with the same issue
- Pydantic documentation: https://docs.pydantic.dev/latest/concepts/alias/#populate-by-name

## Proposed Fix

Add `populate_by_name=True` to the `BaseModelWithConfig` configuration to allow fields with aliases to be populated by either their field name or alias:

```diff
--- a/fastapi/openapi/models.py
+++ b/fastapi/openapi/models.py
@@ -58,7 +58,7 @@ class BaseModelWithConfig(BaseModel):
 class BaseModelWithConfig(BaseModel):
     if PYDANTIC_V2:
-        model_config = {"extra": "allow"}
+        model_config = {"extra": "allow", "populate_by_name": True}

     else:

```