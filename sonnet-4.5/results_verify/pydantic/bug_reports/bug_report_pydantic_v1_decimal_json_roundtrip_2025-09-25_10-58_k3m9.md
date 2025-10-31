# Bug Report: pydantic.v1 Decimal Precision Loss in JSON Roundtrip

**Target**: `pydantic.v1.BaseModel` JSON serialization with `Decimal` fields
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Decimal values lose precision when serialized to JSON via `model.json()` and deserialized via `parse_raw()`, causing silent data corruption.

## Property-Based Test

```python
from decimal import Decimal
from hypothesis import given, strategies as st
import pydantic.v1 as pydantic


@given(st.decimals(allow_nan=False, allow_infinity=False, places=2))
def test_decimal_roundtrip(d):
    class TestModel(pydantic.BaseModel):
        value: Decimal

    model = TestModel(value=d)
    json_repr = model.json()
    reconstructed = TestModel.parse_raw(json_repr)

    assert model.value == reconstructed.value
```

**Failing input**: `Decimal('129452608601646.99')`

## Reproducing the Bug

```python
from decimal import Decimal
import pydantic.v1 as pydantic


class TestModel(pydantic.BaseModel):
    value: Decimal


original_decimal = Decimal('129452608601646.99')
model = TestModel(value=original_decimal)

json_repr = model.json()
reconstructed = TestModel.parse_raw(json_repr)

print(f"Original:      {model.value}")
print(f"Reconstructed: {reconstructed.value}")
print(f"Equal:         {model.value == reconstructed.value}")
print(f"Loss:          {model.value - reconstructed.value}")
```

**Output**:
```
Original:      129452608601646.99
Reconstructed: 129452608601646.98
Equal:         False
Loss:          0.01
```

## Why This Is A Bug

Pydantic claims to support `Decimal` fields for precise numeric handling, but when using JSON serialization (`model.json()` + `parse_raw()`), Decimal values are converted to floats, which have limited precision. This causes silent data corruption - values round-trip through JSON with different values than the original.

This violates the fundamental expectation that serialization/deserialization should be a round-trip operation that preserves data integrity. The issue occurs because JSON natively only supports float numbers, and pydantic serializes Decimals as JSON numbers instead of strings.

Notably, `model.dict()` + `parse_obj()` preserves Decimal precision correctly, showing this is specific to JSON serialization.

## Fix

The fix should serialize Decimal values as strings in JSON to preserve exact precision:

```diff
--- a/pydantic/v1/json.py
+++ b/pydantic/v1/json.py
@@ -XX,X +XX,X @@ def pydantic_encoder(obj: Any) -> Any:
     from decimal import Decimal

     if isinstance(obj, Decimal):
-        return float(obj)
+        return str(obj)
```

The corresponding parser should then convert string representations back to Decimal when deserializing from JSON.