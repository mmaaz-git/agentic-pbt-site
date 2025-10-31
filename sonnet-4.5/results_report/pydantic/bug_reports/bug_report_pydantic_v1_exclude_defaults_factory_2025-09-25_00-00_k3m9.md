# Bug Report: pydantic.v1 exclude_defaults Inconsistent with default_factory

**Target**: `pydantic.v1.BaseModel.dict(exclude_defaults=True)`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

When using `BaseModel.dict(exclude_defaults=True)`, fields with `default_factory` are not excluded even when they have their default value, while fields with regular defaults are correctly excluded. This inconsistent behavior violates the expected semantics of `exclude_defaults`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pydantic.v1 as pv1
from pydantic.v1 import BaseModel, Field


@given(st.text())
@settings(max_examples=500)
def test_exclude_defaults_in_dict(required):
    class ModelWithDefaults(BaseModel):
        required: str
        with_default: int = 42
        with_factory: list = Field(default_factory=list)

    model = ModelWithDefaults(required=required)
    d = model.dict(exclude_defaults=True)

    assert 'required' in d
    assert 'with_default' not in d, "with_default should be excluded when exclude_defaults=True"
    assert 'with_factory' not in d, "with_factory should be excluded when exclude_defaults=True"
```

**Failing input**: Any string value for `required` (e.g., `''`)

## Reproducing the Bug

```python
from pydantic.v1 import BaseModel, Field


class ModelWithDefaults(BaseModel):
    required: str
    regular_default: int = 42
    factory_default: list = Field(default_factory=list)


model = ModelWithDefaults(required='test')

d = model.dict(exclude_defaults=True)

assert 'regular_default' not in d
assert 'factory_default' not in d
```

## Why This Is A Bug

The `exclude_defaults=True` parameter should exclude all fields that have their default value. Currently:
- Fields with regular defaults (e.g., `int = 42`) are correctly excluded
- Fields with `default_factory` (e.g., `list = Field(default_factory=list)`) are NOT excluded, even though they have their default value (`[]`)

This inconsistency violates the principle of least surprise and makes it impossible to reliably exclude all default values when serializing models.

## Fix

The issue is likely in the `_iter` method of BaseModel, which checks whether a field should be excluded. The logic for checking if a field has its default value needs to handle `default_factory` fields correctly.

The fix should ensure that fields with `default_factory` are treated the same as fields with regular defaults when `exclude_defaults=True`. This would involve comparing the current value with the result of calling the `default_factory` (for mutable defaults like `list`, `dict`, etc.).