# Bug Report: Pydantic model_dump_json round_trip Parameter Violates Contract with by_alias=False

**Target**: `pydantic.BaseModel.model_dump_json`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

When using `model_dump_json(by_alias=False, round_trip=True)` on models with field aliases, the output JSON cannot be parsed back into the model, violating the documented contract of the `round_trip` parameter which states "dumped values should be valid as input for the model".

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pydantic import BaseModel, Field

class ModelWithAliasAndRoundTrip(BaseModel):
    field_one: str = Field(alias="fieldOne")
    field_two: int = Field(alias="fieldTwo")

@st.composite
def alias_roundtrip_strategy(draw):
    field_one = draw(st.text(min_size=1, max_size=50))
    field_two = draw(st.integers(min_value=-1000, max_value=1000))
    return ModelWithAliasAndRoundTrip(**{"fieldOne": field_one, "fieldTwo": field_two})

@settings(max_examples=500)
@given(alias_roundtrip_strategy())
def test_round_trip_without_by_alias(model):
    json_str = model.model_dump_json(by_alias=False, round_trip=True)
    restored = ModelWithAliasAndRoundTrip.model_validate_json(json_str)
    assert model.model_dump() == restored.model_dump()
```

**Failing input**: `ModelWithAliasAndRoundTrip(field_one='0', field_two=0)`

## Reproducing the Bug

```python
from pydantic import BaseModel, Field

class ModelWithAlias(BaseModel):
    field_one: str = Field(alias="fieldOne")

model = ModelWithAlias(fieldOne="test")

json_str = model.model_dump_json(by_alias=False, round_trip=True)
print(f"JSON: {json_str}")

restored = ModelWithAlias.model_validate_json(json_str)
```

Output:
```
JSON: {"field_one":"test"}
ValidationError: 1 validation error for ModelWithAlias
fieldOne
  Field required [type=missing, input_value={'field_one': 'test'}, input_type=dict]
```

## Why This Is A Bug

The `model_dump_json` method's documentation explicitly states:
> `round_trip`: If True, dumped values should be valid as input for the model

When `by_alias=False` and `round_trip=True` are used together on a model with field aliases:
1. The JSON output uses internal field names (e.g., `field_one`)
2. The model validation expects alias names (e.g., `fieldOne`)
3. Therefore, the output cannot be parsed back into the model
4. This directly violates the `round_trip` parameter's contract

Note: The issue can be worked around by setting `populate_by_name=True` in the model config, but this doesn't change the fact that `round_trip=True` should guarantee round-trip compatibility regardless of other settings.

## Fix

The fix should ensure that when `round_trip=True`, the serialized output is always valid as input. Possible solutions:

**Option 1**: When `round_trip=True` and `by_alias=False`, temporarily enable field name population during validation:

```diff
--- a/pydantic/main.py
+++ b/pydantic/main.py
@@ -model_dump_json
     def model_dump_json(..., round_trip: bool = False, ...):
+        # When round_trip=True and by_alias=False with aliases,
+        # we need to ensure the output can be parsed back
         ...
+        if round_trip and not by_alias:
+            # Check if model has any aliases
+            # If so, serialize with metadata indicating populate_by_name should be used
```

**Option 2**: Make `round_trip=True` override `by_alias=False` when aliases exist:

```diff
--- a/pydantic/main.py
+++ b/pydantic/main.py
@@ -model_dump_json
     def model_dump_json(..., by_alias: bool = False, round_trip: bool = False, ...):
+        # round_trip takes precedence: ensure output is valid as input
+        if round_trip and has_field_aliases(self):
+            by_alias = True
         ...
```

**Option 3**: Raise a clear error when incompatible parameters are used:

```diff
--- a/pydantic/main.py
+++ b/pydantic/main.py
@@ -model_dump_json
     def model_dump_json(..., by_alias: bool = False, round_trip: bool = False, ...):
+        if round_trip and not by_alias and has_field_aliases(self) and not self.model_config.get('populate_by_name'):
+            raise ValueError(
+                "round_trip=True with by_alias=False requires populate_by_name=True in model config when using field aliases"
+            )
         ...
```

The most user-friendly solution is **Option 2**, as it maintains the `round_trip` guarantee without requiring users to understand the interaction between these parameters.