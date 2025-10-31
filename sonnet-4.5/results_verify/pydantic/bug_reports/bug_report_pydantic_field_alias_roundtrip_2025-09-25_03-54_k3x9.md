# Bug Report: pydantic Field Alias Round-Trip Broken

**Target**: `pydantic.BaseModel.model_dump` and `pydantic.BaseModel.model_validate`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When a model uses field aliases and `model_dump(by_alias=False)` is called, the resulting dictionary cannot be validated with `model_validate()` unless `populate_by_name=True` is set in the model config. This breaks the fundamental round-trip property of serialization.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pydantic import BaseModel, Field


class ModelWithAliases(BaseModel):
    user_name: str = Field(alias='userName')
    user_age: int = Field(alias='userAge')
    is_active: bool = Field(alias='isActive')


@given(
    name=st.text(min_size=0, max_size=100),
    age=st.integers(min_value=-1000000, max_value=1000000),
    active=st.booleans()
)
@settings(max_examples=500)
def test_alias_dump_validate_roundtrip_without_by_alias(name, age, active):
    model = ModelWithAliases(userName=name, userAge=age, isActive=active)

    dumped = model.model_dump(by_alias=False)
    restored = ModelWithAliases.model_validate(dumped)

    assert model == restored
```

**Failing input**: `name='', age=0, active=False` (and all other inputs)

## Reproducing the Bug

```python
from pydantic import BaseModel, Field


class ModelWithAliases(BaseModel):
    user_name: str = Field(alias='userName')
    user_age: int = Field(alias='userAge')


model = ModelWithAliases(userName="Alice", userAge=30)

dumped_no_alias = model.model_dump(by_alias=False)

restored = ModelWithAliases.model_validate(dumped_no_alias)
```

**Output**:
```
ValidationError: 2 validation errors for ModelWithAliases
userName
  Field required [type=missing, input_value={'user_name': 'Alice', 'user_age': 30}, input_type=dict]
userAge
  Field required [type=missing, input_value={'user_name': 'Alice', 'user_age': 30}, input_type=dict]
```

## Why This Is A Bug

The round-trip property is fundamental to serialization systems: `validate(dump(model)) == model`. This property is violated when field aliases are used without `populate_by_name=True`.

When `by_alias=False` is specified in `model_dump()`, it returns the Python field names (e.g., `user_name`), but `model_validate()` only accepts the alias names (e.g., `userName`) by default. This inconsistency breaks the round-trip property.

## Fix

The fix should make `model_validate()` accept both field names and aliases by default, or at minimum, document this behavior clearly and provide a parameter to `model_validate()` to control this behavior.

A workaround exists: set `populate_by_name=True` in the model config:

```python
class ModelWithAliases(BaseModel):
    model_config = {'populate_by_name': True}

    user_name: str = Field(alias='userName')
    user_age: int = Field(alias='userAge')
```

This allows `model_validate()` to accept both Python field names and alias names, fixing the round-trip property. However, this should be the default behavior or the `by_alias` parameter should be consistent across dump and validate operations.