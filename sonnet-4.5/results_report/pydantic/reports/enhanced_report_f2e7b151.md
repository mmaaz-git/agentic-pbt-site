# Bug Report: pydantic.BaseModel.model_dump_json Violates Round-Trip Contract with Field Aliases

**Target**: `pydantic.BaseModel.model_dump_json`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `model_dump_json` method violates its documented contract when `round_trip=True` and `by_alias=False` are used together on models with field aliases. The output JSON cannot be parsed back into the model, contradicting the round_trip parameter's promise that "dumped values should be valid as input for the model".

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

if __name__ == "__main__":
    test_round_trip_without_by_alias()
```

<details>

<summary>
**Failing input**: `ModelWithAliasAndRoundTrip(field_one='0', field_two=0)`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 22, in <module>
    test_round_trip_without_by_alias()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 15, in test_round_trip_without_by_alias
    @given(alias_roundtrip_strategy())
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 18, in test_round_trip_without_by_alias
    restored = ModelWithAliasAndRoundTrip.model_validate_json(json_str)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pydantic/main.py", line 656, in model_validate_json
    return cls.__pydantic_validator__.validate_json(json_data, strict=strict, context=context)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
pydantic_core._pydantic_core.ValidationError: 2 validation errors for ModelWithAliasAndRoundTrip
fieldOne
  Field required [type=missing, input_value={'field_one': '0', 'field_two': 0}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.10/v/missing
fieldTwo
  Field required [type=missing, input_value={'field_one': '0', 'field_two': 0}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.10/v/missing
Falsifying example: test_round_trip_without_by_alias(
    model=ModelWithAliasAndRoundTrip(field_one='0', field_two=0),  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from pydantic import BaseModel, Field

class ModelWithAlias(BaseModel):
    field_one: str = Field(alias="fieldOne")

model = ModelWithAlias(fieldOne="test")

json_str = model.model_dump_json(by_alias=False, round_trip=True)
print(f"JSON: {json_str}")

try:
    restored = ModelWithAlias.model_validate_json(json_str)
    print(f"Successfully restored: {restored}")
except Exception as e:
    print(f"Error: {e}")
```

<details>

<summary>
ValidationError: Field 'fieldOne' required but got 'field_one' in JSON
</summary>
```
JSON: {"field_one":"test"}
Error: 1 validation error for ModelWithAlias
fieldOne
  Field required [type=missing, input_value={'field_one': 'test'}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.10/v/missing
```
</details>

## Why This Is A Bug

This violates the explicit contract of the `round_trip` parameter. According to Pydantic's documentation, when `round_trip=True`, the "dumped values should be valid as input for the model". However, when combined with `by_alias=False` on models with field aliases:

1. **Serialization uses internal field names**: With `by_alias=False`, `model_dump_json` outputs the internal field names (e.g., `"field_one"` instead of `"fieldOne"`).

2. **Deserialization expects aliases**: By default, Pydantic models with field aliases only accept the alias names during validation, not the internal field names.

3. **Contract violation**: The JSON produced by `model_dump_json(by_alias=False, round_trip=True)` cannot be parsed back using `model_validate_json()`, directly violating the documented guarantee that the output should be "valid as input".

4. **No documented warnings**: The documentation does not warn about this incompatibility or provide any caveats about using these parameters together.

The issue is deterministic and affects any model with field aliases when these specific parameters are combined, making it a clear violation of the method's contract rather than an edge case.

## Relevant Context

- **Pydantic Version**: 2.10.3
- **Python Version**: 3.13
- **Documentation Reference**: The `model_dump_json` method documentation states that `round_trip` should make "dumped values valid as input for the model"
- **Workaround Available**: Setting `populate_by_name=True` in the model's configuration allows the model to accept both field names and aliases during validation, working around this issue
- **Common Use Case**: Field aliases are extensively used in Pydantic for API integrations where Python naming conventions differ from external systems (e.g., snake_case vs camelCase)

Related Pydantic documentation:
- [Field aliases](https://docs.pydantic.dev/latest/concepts/fields/#field-aliases)
- [Serialization](https://docs.pydantic.dev/latest/concepts/serialization/)

## Proposed Fix

The most user-friendly solution is to make `round_trip=True` take precedence over `by_alias=False` when the model has field aliases, ensuring the contract is always honored:

```diff
--- a/pydantic/main.py
+++ b/pydantic/main.py
@@ -model_dump_json
     def model_dump_json(
         self,
         *,
         indent: int | None = None,
         include: IncEx | None = None,
         exclude: IncEx | None = None,
         context: Any | None = None,
         by_alias: bool = False,
         exclude_unset: bool = False,
         exclude_defaults: bool = False,
         exclude_none: bool = False,
         round_trip: bool = False,
         warnings: bool | Literal['none', 'warn', 'error'] = True,
         serialize_unknown: bool = False,
     ) -> str:
+        # When round_trip=True, ensure output can be parsed back
+        # by using aliases if the model has any field aliases
+        if round_trip and not by_alias:
+            for field_name, field_info in self.model_fields.items():
+                if field_info.alias and field_info.alias != field_name:
+                    # Model has aliases, must use them for round-trip
+                    by_alias = True
+                    break
+
         return self.__pydantic_serializer__.to_json(
             self,
             indent=indent,
             include=include,
             exclude=exclude,
             context=context,
             by_alias=by_alias,
             exclude_unset=exclude_unset,
             exclude_defaults=exclude_defaults,
             exclude_none=exclude_none,
             round_trip=round_trip,
             warnings=warnings,
             serialize_unknown=serialize_unknown,
         ).decode()
```